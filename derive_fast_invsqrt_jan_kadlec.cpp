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

template <size_t N, typename T>
static inline T get_nth_power(T x) {
    if constexpr(N == 0) return T(1);
    if constexpr(N == 1) return x;
    T y = get_nth_power<N/2, T>(x);
    T y_2 = y*y;
    if constexpr(N % 2 == 0) return y_2;
    return x*y_2;
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
        rand_params.k1 = rng_f32(rng_gen)*-5.0;
        rand_params.k2 = rng_f32(rng_gen)*5.0;
        // Based on Jan Kadlec's version
        // rand_params.k0 = M0 - 0x5F1F'FFF9;
        // rand_params.k1 = -0.703952253;
        // rand_params.k2 = 0.703952253 * 2.38924456;
        return rand_params;
    };

    struct Result {
        Params params;
        double loss = double(~uint64_t(0));
        size_t version = 0;
        size_t iter = 0;
        size_t thread_id = 0;
    };

    Result best_result;
    std::mutex best_result_mutex;
    constexpr size_t MAXIMUM_PLATEAU_RESTART = 1'000;
    constexpr size_t IS_PRINT_ITER = false;
    constexpr size_t PRINT_ITER = 100'000;

    std::vector<std::thread> threads;
    const size_t TOTAL_THREADS = std::thread::hardware_concurrency();
    for (size_t thread_id = 0; thread_id < TOTAL_THREADS; thread_id++) {
        threads.push_back(std::thread([&, thread_id]() {
            // NOTE: Increase P_norm to trade off lower mean absolute error for lower max absolute error
            constexpr size_t P_norm = 6; // minimise max absolute error
            constexpr double P_norm_scale = 1e2;
            constexpr double scale_de_dk[3] = {1e4, 1e0, 1e0};
            // constexpr size_t P_norm = 2; // mean squared error
            // constexpr double P_norm_scale = 1e2;
            // constexpr double scale_de_dk[3] = {1e3, 1e0, 1e0};
            // constexpr size_t P_norm = 1; // mean absolute error
            // constexpr double P_norm_scale = 1e0;
            // constexpr double scale_de_dk[3] = {1e-1, 1e-3, 1e-3};

            Result best_thread_result;
            Result best_ver_result;
            size_t total_plateau = 0;
            size_t curr_version = 0;
            size_t curr_iter = 0;
            Params curr_params;
            curr_params = gen_rand_params();

            while (is_running) {
                double mean_loss = 0;
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
                    // e = 1/n*(1 - y1/y_target)^n
                    // e = 1/n*(1- g(y0)/y_target)^n
                    // de/dg = -1/y_target * (y_target - g(y0))^(n-1)
                    const double y_f64 = double(y_f32);
                    const double error = (1.0 - y1/y_f64)*P_norm_scale;
                    double de_dg;
                    double loss;
                    if constexpr(P_norm > 1) {
                        de_dg = get_nth_power<P_norm-1>(error);
                        loss = get_nth_power<P_norm>(error);
                        de_dg /= y_f64;
                    } else {
                        // L1 norm is absolute value
                        // Since y = |x| is not differentiable we use pseudo Huber loss function
                        // https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function
                        // y = sqrt(x^2+k^2)-k ~ |x| as k -> 0
                        // dy/dx = x/sqrt(x^2+k^2)
                        // x = 1-g(y0)/y_target
                        // dx/dg = -1/y_target
                        // dy/dg = x/sqrt(x^2+k^2) * -1/y_target
                        constexpr double k_huber = 0.01;
                        constexpr double k_huber_2 = k_huber*k_huber;
                        const double error_2 = get_nth_power<2>(error);
                        de_dg = error/std::sqrt(error_2 + k_huber_2);
                        de_dg /= y_f64;
                        loss = std::abs(error);
                    }
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
                    const double df_dk0 = y0;
                    // chain rule
                    const double de_dk2 = de_dg*dg_dk2;
                    const double de_dk1 = de_dg*dg_dk1;
                    const double de_dk0 = de_dg*dg_df*df_dk0;
                    avg_de_dk[0] += de_dk0;
                    avg_de_dk[1] += de_dk1;
                    avg_de_dk[2] += de_dk2;
                    mean_loss += loss;
                }
                mean_loss /= double(N);

                if (mean_loss < best_ver_result.loss) {
                    best_ver_result.loss = mean_loss;
                    best_ver_result.version = curr_version;
                    best_ver_result.iter = curr_iter;
                    best_ver_result.params = curr_params;
                    best_ver_result.thread_id = thread_id;
                    total_plateau = 0;
                } else {
                    total_plateau++;
                }

                if (best_ver_result.loss < best_thread_result.loss) {
                    best_thread_result = best_ver_result;
                }

                const bool is_reset = (total_plateau >= MAXIMUM_PLATEAU_RESTART);
                // higher order polynomials have problems with vanishing gradients as loss improves
                // so we compensate by rescaling the gradients by the loss function to counteract this
                const double loss_rescale = std::abs(mean_loss);
                for (size_t i = 0; i < 3; i++) {
                    const double scale = scale_de_dk[i]/double(N);
                    avg_de_dk[i] *= scale;
                    avg_de_dk[i] /= loss_rescale;
                }
                curr_params.k0 += int64_t(avg_de_dk[0]);
                curr_params.k1 += avg_de_dk[1];
                curr_params.k2 += avg_de_dk[2];
                if (is_reset || (IS_PRINT_ITER && (curr_iter % PRINT_ITER == 0))) {
                    printf(
                        "thread=%.2zu, version=%.4zu, iter=%.10zu, loss=%.3e, de_dk[3]={%.2e,%.2e,%.2e}\n",
                        thread_id, curr_version, curr_iter,
                        best_ver_result.loss,
                        avg_de_dk[0], avg_de_dk[1], avg_de_dk[2]
                    );
                }
                if (is_reset) {
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
            if (best_thread_result.loss < best_result.loss) {
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
    printf("loss = %.3e\n", best_result.loss);
    printf("total_samples = %zu\n", N);
    // k0_ = 2^22*381 - k0
    const int64_t k0_ = M0 - best_result.params.k0;
    printf("k0 = %" PRIi64 "\n", k0_);
    printf("k1 = %.8f\n", best_result.params.k1);
    printf("k2 = %.8f\n", best_result.params.k2);

    return 0;
}
