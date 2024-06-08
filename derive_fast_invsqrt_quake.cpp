#include <cmath>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <vector>

static inline int32_t quantise_float(float x) {
    union {
        float f32;
        int32_t i32;
    } y;
    y.f32 = x;
    return y.i32;
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
    // Qy ~ 2^23*(190.5 - 1.5*k0') - 0.5*Qx
    // Qy ~ 2^22*381 - 2^22*3*k0' - 0.5*Qx
    // Qy ~ 2^22*381 - k1 - 0.5*Qx
    constexpr int64_t M0 = int64_t(381) << 22;

    std::vector<int64_t> QX;
    std::vector<int64_t> QY_target;
    const auto push_sample = [&](float x) {
        const float y = 1.0f/std::sqrt(x);
        const int32_t Qx = quantise_float(x);
        const int32_t Qy = quantise_float(y);
        QX.push_back(int64_t(Qx));
        QY_target.push_back(int64_t(Qy));
    };

    // Relative error is periodic so we only need to sample a single decade
    for (float x = 1e-1f; x <= 1e0f; x += 1e-3f) push_sample(x);
    const size_t N = QX.size();
    printf("training over %zu samples\n", N);

    // Increase P_norm to trade off lower mean squared error for lower max absolute error
    constexpr size_t P_norm = 6; // minimise maximum absolute error like original Quake
    constexpr double P_norm_scale = 4e4;
    constexpr double grad_scale = 1e6;
    // constexpr size_t P_norm = 4;
    // constexpr double P_norm_scale = 1e5;
    // constexpr double grad_scale = 1e7;
    // constexpr size_t P_norm = 2; // mean squared error
    // constexpr double P_norm_scale = 1e8;
    // constexpr double grad_scale = 1e8;
    // constexpr size_t P_norm = 1; // mean absolute error
    // constexpr double P_norm_scale = 1e12;
    // constexpr double grad_scale = 1e11;

    int64_t k1 = 0;
    int64_t best_k1 = k1;
    double best_loss = double(~uint64_t(0));
    size_t best_iter = 0;
    constexpr size_t TOTAL_ITERATIONS = 102400;
    constexpr size_t PRINT_ITER = TOTAL_ITERATIONS / 32;
    for (size_t iter = 0; iter < TOTAL_ITERATIONS; iter++) {
        double mean_de_dk = 0;
        double mean_loss = 0;
        for (size_t i = 0; i < N; i++) {
            // Qy ~ 2^22*381 - k1 - 0.5*Qx
            const int64_t Qx = QX[i];
            const int64_t Qy_target = QY_target[i];
            const int64_t Qy_pred = M0 - k1 - (Qx >> 1); // Qx > 0
            // error = 1/n*(1 - Qy/Qy')^n
            // de/dk = (1-Qy/Qy')^(n-1) * 1/Qy'
            const double Qy_target_f64 = double(Qy_target);
            const double error = (1.0 - double(Qy_pred)/Qy_target_f64)*P_norm_scale;
            double de_dk;
            double loss;
            if constexpr(P_norm > 1) {
                de_dk = get_nth_power<P_norm-1>(error);
                loss = get_nth_power<P_norm>(error);
                de_dk /= Qy_target_f64;
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
                de_dk = error/std::sqrt(error_2 + k_huber_2);
                de_dk /= Qy_target_f64;
                loss = std::abs(error);
            }
            mean_de_dk += de_dk;
            mean_loss += loss;
        }
        mean_de_dk /= double(N);
        mean_loss /= double(N);

        if (mean_loss < best_loss) {
            best_loss = mean_loss;
            best_k1 = k1;
            best_iter = iter;
        }

        const int64_t gradient = -int64_t(mean_de_dk*grad_scale);
        k1 += gradient;
        if (iter % PRINT_ITER == 0 || gradient == 0) {
            printf("iter=%.3zu, loss=%.3e, mean_de_dk=%.3e, gradient=%" PRIi64 "\n", 
                iter, mean_loss, mean_de_dk, gradient);
        }
        if (gradient == 0) {
            break;
        }
    }

    printf("\n[BEST RESULTS]\n");
    printf("p-norm = %zu\n", P_norm);
    printf("iter = %zu\n", best_iter);
    printf("loss = %.3e\n", best_loss);
    printf("k1 = %" PRIi64 "\n", best_k1);
    // k1 = 2^22*3*k0'
    const double k0_ = double(best_k1)/double(int64_t(3) << 22);
    printf("k' = %.8f\n", k0_);
    // k0 = 2^22*381 - k1
    const int32_t k0 = int32_t(M0 - best_k1);
    printf("k0 = %" PRIi32 "\n", k0);
    // Qy = k0 - 0.5*Qx

    return 0;
}
