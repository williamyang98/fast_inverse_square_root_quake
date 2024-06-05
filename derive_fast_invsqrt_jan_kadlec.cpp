#include <cmath>
#include <inttypes.h>
#include <mutex>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <thread>
#include <vector>

#define USE_SIGNAL_HANDLER 1

#if USE_SIGNAL_HANDLER
static bool volatile is_running = true;
#if _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
BOOL WINAPI sighandler(DWORD signum) {
    if (signum == CTRL_C_EVENT) {
        fprintf(stderr, "Signal caught, exiting!\n");
        is_running = false;
        return TRUE;
    }
    return FALSE;
}
#else
#include <errno.h>
#include <signal.h>
static void sighandler(int signum) {
    fprintf(stderr, "Signal caught, exiting! (%d)\n", signum);
    is_running = false;
}
#endif
#endif

static inline int32_t quantise_float(float x) {
    union {
        float f32;
        int32_t i32;
    } y;
    y.f32 = x;
    return y.i32;
}

static inline float dequantise_i32(int32_t x) {
    union {
        float f32;
        int32_t i32;
    } y;
    y.i32 = x;
    return y.f32;
}

// clang++ main.cpp -o main -std=c++17 -O3 -march=native
int main(int argc, char** argv) {
#if USE_SIGNAL_HANDLER
#if _WIN32
    SetConsoleCtrlHandler(sighandler, TRUE);
#else
    struct sigaction sigact;
    sigact.sa_handler = sighandler;
    sigemptyset(&sigact.sa_mask);
    sigact.sa_flags = 0;
    sigaction(SIGINT, &sigact, nullptr);
    sigaction(SIGTERM, &sigact, nullptr);
    sigaction(SIGQUIT, &sigact, nullptr);
    sigaction(SIGPIPE, &sigact, nullptr);
#endif
#endif

    std::vector<float> X_in;
    std::vector<float> Y_target;
    std::vector<int32_t> Qy_target;
    const auto push_sample = [&](float x) {
        const float y = 1.0f/std::sqrt(x);
        X_in.push_back(x);
        Y_target.push_back(y);
        Qy_target.push_back(quantise_float(y));
    };

    // Need to train over a single decade since relative error is periodic
    for (float x = 1e-1f; x <= 1e0f; x += 1e-3f) push_sample(x);
    const size_t N = X_in.size();
    printf("training over %zu samples\n", N);

    // Solve for the following approximation
    // Qy ~ 2^22*381 - k0 - 0.5*Qx, k0' = 2^22*381 - k0
    // y0 = dequant(Qy)
    // y1 = y0*(k1*x*y0^2 + k2) | Custom Newton's method
    constexpr int64_t M0 = int64_t(381) << 22;
    const double M1 = std::log(double(2.0))/double(int32_t(3) << 22);
    struct Params {
        int64_t k0 = 0;
        double k1 = 0.0;
        double k2 = 0.0;
    };

    std::random_device rng_dev;
    std::mt19937 rng_gen(rng_dev());
    std::uniform_real_distribution<double> rng_f32(0.0, 1.0);
    const auto gen_rand_params = [&]() -> Params {
        Params rand_params;
        rand_params.k0 = int64_t(rng_f32(rng_gen)*1e7);
        rand_params.k1 = rng_f32(rng_gen)*-2.0;
        rand_params.k2 = rng_f32(rng_gen)*2.0;
        // Based on Jan Kadlec's version
        // rand_params.k0 = M0 - 0x5F1F'FFF9;
        // rand_params.k1 = -0.703952253;
        // rand_params.k2 = 0.703952253 * 2.38924456;
        return rand_params;
    };

    struct Result {
        Params params;
        double mae = double(~uint64_t(0));
        size_t version = 0;
        size_t iter = 0;
        size_t thread_id = 0;
    };

    Result best_result;
    std::mutex best_result_mutex;
    constexpr size_t PRINT_ITER = 20'000;
    constexpr size_t MAXIMUM_PLATEAU_RESTART = 1'000;

    std::vector<std::thread> threads;
    const size_t TOTAL_THREADS = std::thread::hardware_concurrency();
    for (size_t thread_id = 0; thread_id < TOTAL_THREADS; thread_id++) {
        threads.push_back(std::thread([&, thread_id]() {
            Result best_thread_result;
            Result best_ver_result;
            size_t total_plateau = 0;
            size_t curr_version = 0;
            size_t curr_iter = 0;
            Params curr_params;
            curr_params = gen_rand_params();

            while (is_running) {
                double mean_absolute_error = 0;
                double avg_de_dk[3] = {0};
                for (size_t i = 0; i < N; i++) {
                    const float x_f32 = X_in[i];
                    const float y_f32 = Y_target[i];
                    // Qx ~ 2^23 * [log2(x) + 127 - k0']
                    const int32_t Qx = quantise_float(x_f32);
                    // Qy ~ 2^23*(190.5 - 1.5*k0') - 0.5*Qx
                    // Qy ~ 2^22*(381 - 3*k0') - 0.5*Qx
                    // Qy ~ 2^22*381 - k0 - 0.5*Qx, where k0 = 2^22*3*k0'
                    const int64_t Qy = M0 - curr_params.k0 - (Qx >> 1);
                    // Qy ~ 2^23 * [log2(y) + 127 - k0']
                    // log2(y) ~ Qy*2^-23 - 127 + k0'
                    // y = f(Qy) ~ 2^[ Qy*2^-23 - 127 + k0/(3*2^22) ]
                    const double y0 = double(dequantise_i32(int32_t(Qy)));
                    // y1 = g(y0) = y0*(k1*x*y0^2 + k2)
                    const double x_f64 = double(x_f32);
                    const double y1 = y0*(curr_params.k1*x_f64*y0*y0 + curr_params.k2);
                    // e = 0.5*(y_target - y1)^2
                    // e = 0.5*(y_target - g(y0))^2
                    // de/dg = -(y_target - g(y0))
                    const double y_f64 = double(y_f32);
                    const double error = 1.0 - y1/y_f64;
                    const double de_dg = error/y_f64;
                    // g(y0) = y0*(k1*x*y0^2 + k2)
                    // dg/dk1 = x*y0^3
                    // dg/dk2 = y0
                    const double dg_dk1 = x_f64*y0*y0*y0;
                    const double dg_dk2 = y0;
                    // g(y0) = g(f(Qy)), y0 = f(Qy)
                    // g(Qy) = f(Qy)*(k1*x*f(Qy)^2 + k2)
                    // dg/df = 3*k1*x*f(Qy)^2 + k2
                    const double dg_df = 3.0*curr_params.k1*x_f64*y0*y0 + curr_params.k2;
                    // f(Qy) ~ 2^[ Qy*2^-23 - 127 + k0/(3*2^22) ]
                    // df/dk0 ~ ln(2)/(3*2^22) * f(Qy)
                    const double df_dk0 = M1 * y0;
                    // chain rule
                    const double de_dk2 = de_dg*dg_dk2;
                    const double de_dk1 = de_dg*dg_dk1;
                    const double de_dk0 = de_dg*dg_df*df_dk0;
                    avg_de_dk[0] += de_dk0;
                    avg_de_dk[1] += de_dk1;
                    avg_de_dk[2] += de_dk2;
                    mean_absolute_error += std::abs(error);
                }
                mean_absolute_error /= double(N);

                if (mean_absolute_error < best_ver_result.mae) {
                    best_ver_result.mae = mean_absolute_error;
                    best_ver_result.version = curr_version;
                    best_ver_result.iter = curr_iter;
                    best_ver_result.params = curr_params;
                    best_ver_result.thread_id = thread_id;
                    total_plateau = 0;
                } else {
                    total_plateau++;
                }

                if (best_ver_result.mae < best_thread_result.mae) {
                    best_thread_result = best_ver_result;
                }

                const bool is_reset = (total_plateau >= MAXIMUM_PLATEAU_RESTART);

                constexpr static double scale_de_dk[3] = {1e-1, 1e-1, 1e-1};
                for (size_t i = 0; i < 3; i++) {
                    const double scale = scale_de_dk[i]/double(N);
                    avg_de_dk[i] *= scale;
                }
                curr_params.k0 += int64_t(avg_de_dk[0]);
                curr_params.k1 += avg_de_dk[1];
                curr_params.k2 += avg_de_dk[2];
                if (is_reset) {
                    printf(
                        "thread=%.2zu, version=%.4zu, iter=%.10zu, mae=%.3e, de_dk[3]={%.2e,%.2e,%.2e}\n",
                        thread_id, curr_version, curr_iter,
                        best_ver_result.mae,
                        avg_de_dk[0], avg_de_dk[1], avg_de_dk[2]
                    );
                    total_plateau = 0;
                    curr_version++;
                    curr_iter = 0;
                    curr_params = gen_rand_params();
                    best_ver_result = Result{};
                    continue;
                }
                curr_iter++;
            }
            auto lock = std::unique_lock<std::mutex>(best_result_mutex);
            if (best_thread_result.mae < best_result.mae) {
                best_result = best_thread_result;
            }
        }));
    }

    for (auto& thread: threads) {
        thread.join();
    }

    printf("\n[BEST RESULT]\n");
    printf("thread = %zu\n", best_result.thread_id);
    printf("version = %zu\n", best_result.version);
    printf("iter = %zu\n", best_result.iter);
    printf("mae = %.3e\n", best_result.mae);
    printf("total_samples = %zu\n", N);
    // k0_ = 2^22*381 - k0
    const int64_t k0_ = M0 - best_result.params.k0;
    printf("k0 = %" PRIi64 "\n", k0_);
    printf("k1 = %.8f\n", best_result.params.k1);
    printf("k2 = %.8f\n", best_result.params.k2);

    return 0;
}
