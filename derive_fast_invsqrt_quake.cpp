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

// clang++ main.cpp -o main -std=c++17 -O3 -march=native
int main(int argc, char** argv) {
    // Qy ~ 2^23*(190.5 - 1.5*k0') - 0.5*Qx
    // 2*Qy ~ 2^23*381 - 2^23*3*k0' - Qx
    // Jy ~ 2^23*381 - 2^23*3*k0' - Qx
    // Jy ~ 2^23*381 - k1 - Qx
    constexpr int64_t M0 = int64_t(381) << 23;

    std::vector<int64_t> QX;
    std::vector<int64_t> JY_target;
    const auto push_sample = [&](float x) {
        const float y = 1.0f/std::sqrt(x);
        const int32_t Qx = quantise_float(x);
        const int32_t Qy = quantise_float(y);
        const int64_t Jy = int64_t(Qy)*2;
        QX.push_back(int64_t(Qx));
        JY_target.push_back(Jy);
    };

    for (float x = 1e-7f; x <= 1e-6f; x += 1e-9f) push_sample(x);
    for (float x = 1e-6f; x <= 1e-5f; x += 1e-8f) push_sample(x);
    for (float x = 1e-5f; x <= 1e-4f; x += 1e-7f) push_sample(x);
    for (float x = 1e-4f; x <= 1e-3f; x += 1e-6f) push_sample(x);
    for (float x = 1e-3f; x <= 1e-2f; x += 1e-5f) push_sample(x);
    for (float x = 1e-2f; x <= 1e-1f; x += 1e-4f) push_sample(x);
    for (float x = 1e-1f; x <= 1e0f; x += 1e-3f) push_sample(x);
    for (float x = 1e0f; x <= 1e1f; x += 1e-2f) push_sample(x);
    for (float x = 1e1f; x <= 1e2f; x += 1e-1f) push_sample(x);
    for (float x = 1e2f; x <= 1e3f; x += 1e0f) push_sample(x);
    for (float x = 1e3f; x <= 1e4f; x += 1e1f) push_sample(x);
    for (float x = 1e4f; x <= 1e5f; x += 1e2f) push_sample(x);
    for (float x = 1e5f; x <= 1e6f; x += 1e3f) push_sample(x);
    for (float x = 1e6f; x <= 1e7f; x += 1e4f) push_sample(x);
    for (float x = 1e7f; x <= 1e8f; x += 1e5f) push_sample(x);

    const size_t N = JY_target.size();
    printf("training over %zu samples\n", N);

    int64_t k1 = 0;
    int64_t best_k1 = k1;
    uint64_t best_mse = ~uint64_t(0);
    size_t best_iter = 0;
    constexpr size_t TOTAL_ITERATIONS = 1024;
    constexpr size_t PRINT_ITER = TOTAL_ITERATIONS / 32;
    for (size_t iter = 0; iter < TOTAL_ITERATIONS; iter++) {
        int64_t mean_error = 0;
        uint64_t mean_square_error = 0;
        for (size_t i = 0; i < N; i++) {
            // Jy ~ 2^23*381 - k1 - Qx
            // error = 0.5*(Jy - Jy')^2
            // error = 0.5*(2^23*381 - k1 - Qx - Qy')^2
            // de/dk = -(Qy-Qy')
            const int64_t Qx = QX[i];
            const int64_t Jy_target = JY_target[i];
            const int64_t Jy_pred = M0 - k1 - Qx;
            const int64_t error = Jy_pred - Jy_target;
            mean_error += error;
            mean_square_error += uint64_t(error*error);
        }
        mean_error /= int64_t(N);
        mean_square_error /= int64_t(N);

        if (mean_square_error < best_mse) {
            best_mse = mean_square_error;
            best_k1 = k1;
            best_iter = iter;
        }

        int64_t gradient = mean_error;
        k1 += gradient;
        if (iter % PRINT_ITER == 0 || mean_error == 0) {
            printf("[%.3zu] mse=%" PRIu64 ", mean_error=%" PRIi64 "\n", iter, mean_square_error, mean_error);
        }
        if (mean_error == 0) {
            break;
        }
    }

    printf("\n[BEST RESULTS]\n");
    printf("iter = %zu\n", best_iter);
    printf("mse = %" PRIu64 "\n", best_mse);
    printf("k1 = %" PRIi64 "\n", best_k1);
    // k1 = 2^23 * 3 * k0'
    const double k0_ = double(best_k1)/double(int64_t(3) << 23);
    printf("k' = %.8f\n", k0_);
    // k0_ = (2^23*381 - k1) / 2
    const int32_t k0 = int32_t((M0 - best_k1)/2);
    printf("k0 = %" PRIi32 "\n", k0);
    // Qy = k0 - 0.5*Qx

    return 0;
}
