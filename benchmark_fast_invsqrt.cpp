#include <cmath>
#include <stdint.h>
#include <stdio.h>
#include <functional>
#include <vector>

float quick_invsqrt_quake(float x, int32_t k0, size_t total_iterations) {
    union {
        int32_t log_val;
        float f32_val;
    } x_pun;
    x_pun.f32_val = x;
    // IEEE-754-1985 format gives the following bit expression for single precision floating point format
    // - number of bits: K: 0, N: 8, M: 23
    // - interpreted as f32
    // x = (-1)^K * 2^(N-127) * (1 + M*2^-23)
    //
    // Quantisation as integer through punning gives
    // - Q = K*2^31 + N*2^23 + M
    // Assume positive value for x, so K = 0 since y=1/sqrt(x) must have x > 0
    // - Q = N*2^23 + M
    //
    // Consider the expansion of log2(x) in terms of IEEE-754
    // x = 2^(N-127) * (1+M*2^-23)
    // - let n = N-127
    //       m = M*2^-23
    // x = 2^n * (1+m)
    // log2(x) = log2[2^n * (1+m)]
    // log2(x) = log2(2^n) + log2(1+m)
    // log2(x) = n + log2(1+m)
    //
    // Since m = M*2^-23 
    // - 0 <= M <= 2^23-1
    // - 0 <= m < 1
    // - 1 <= 1+m < 2
    // - 0 <= log2(1+m) < 1
    //
    // We can approximate log2(1+m) ~ m + k0 (Graphing it's shows it's a pretty good linear approximation)
    // - k0 is a free parameter
    // - k0 = 0 is the solution to fit the endpoints log2(1)=0, log2(2)=1
    // - k0 can be varied to be non-zero to minimize total error across the approximation
    // log2(1+m) ~ m + k0
    //
    // Substituting approximation for log2(1+m)
    // log2(x) = n + log2(1+m)
    // log2(x) ~ n + m + k0
    // - Substitute n and m
    // log2(x) ~ N-127 + M*2^-23 + k0
    // log2(x)*2^23 ~ N*2^23 - 127*2^23 + k0*2^23 + M
    // - Substitute quantisation Q = N*2^23 + M
    // log2(x)*2^23 ~ Q - 127*2^23 + k0*2^23
    // Q ~ 2^23 * [ log2(x) + 127 - k0 ]

    // Deriving expression for y = 1/sqrt(x) in terms of log2
    // log2(y) = log2(x^-0.5)
    // log2(y) = -0.5*log2(x)
    //
    // Using the approximation of our quantisation Q
    // Qy ~ 2^23 * [ log2(y) + 127 - k0 ]
    // Qy ~ 2^23 * [-0.5*log2(x) + 127 - k0 ]
    //
    // Qx ~ 2^23 * [ log2(x) + 127 - k0 ]
    // -0.5*Qx ~ 2^23 * [ -0.5*log2(x) - 63.5 + 0.5*k0 ]
    // -0.5*Qx ~ 2^23 * [ -0.5*log2(x) + 127 - k0 - 190.5 + 1.5*k0]
    // -0.5*Qx ~ 2^23 * [ -0.5*log2(x) + 127 - k0 ] - 2^23 * [ 190.5 - 1.5*k0 ]
    // 
    // Subsituting Qy
    // -0.5*Qx ~ Qy - 2^23 * [190.5 - 1.5*k0]
    // Qy ~ 2^23*(190.5 - 1.5*k0) - 0.5*Qx
    //
    // Let k1 ~ 2^23*(190.5 - 1.5*k0) be our new free parameter
    // Qy ~ k1 - 0.5*Qx
    x_pun.log_val = k0 - (x_pun.log_val >> 1); // Qy ~ k2 - 0.5*Qx
    float y_approx = x_pun.f32_val;

    // Consider f(y) = 1/y^2 - x' = (1 - x'*y^2)/y^2
    // - x' = initial value of x
    // Solving for f(y) = 0, gives y = 1/sqrt(x')
    // Use Newton's method for finding roots, given an initial close guess for y
    // f'(y) = = -2*y^-3 = -2/y^3
    // y1 = y0 - f(y0)/f'(y0)  | y1 = successive approximation
    // - f(y)/f'(y) = f_delta * dy/df = y_delta (y_delta, f_delta relative to closest root)
    // - f(y)/f'(y) = -0.5*y*[1-x'*y^2]
    // y1 = y0 + 0.5*y0*[1 - x'y0^2]
    // y1 = y0 + 0.5*y0 - 0.5*x'*y0^3
    // y1 = 1.5*y0 - 0.5*x'*y0^3
    // y1 = y0*[1.5 - 0.5*x'*y0^2]
    const float x_half = x*0.5f;
    for (size_t i = 0; i < total_iterations; i++) {
        y_approx = y_approx*(1.5f - x_half*y_approx*y_approx);
    }
    return y_approx;
}

struct JanKadlec {
    int32_t k0;
    float k1;
    float k2;
};

float quick_invsqrt_jan_kadlec(float x, JanKadlec params) {
    // Qx = quant(x)
    // Qy ~ k0 - 0.5*Qx
    // y0 = dequant(Qy) 
    // y1 = y0*(k1*x*y0^2 + k2)
    union {
        int32_t log_val;
        float f32_val;
    } x_pun;
    x_pun.f32_val = x;
    x_pun.log_val = params.k0 - (x_pun.log_val >> 1);
    float y = x_pun.f32_val;
    y = y*(params.k1*x*y*y + params.k2);
    return y;
}


// clang++ main.cpp -o main -O3
int main(int argc, char** argv) {
    struct Benchmark {
        const char* name;
        std::function<float(float)> invsqrt;
    };

    std::vector<Benchmark> benchmarks;
    const auto add_quake_benchmark = [&](const char* name, int32_t k0) {
        benchmarks.push_back({ name, [k0](float x) { return quick_invsqrt_quake(x, k0, 1); } });
    };
    add_quake_benchmark("Naive (k0=0)", int32_t(381) << 22);
    add_quake_benchmark("Original Quake", 0x5F3759DF);
    add_quake_benchmark("Gradient descent single parameter", 1597311293);
    benchmarks.push_back({ "Jan Kadlec", 
        [](float x) { 
            JanKadlec params;
            params.k0 = 0x5F1FFFF9;
            params.k1 = -0.703952253f;
            params.k2 = 0.703952253f * 2.38924456f;
            return quick_invsqrt_jan_kadlec(x, params);
        } 
    });
    benchmarks.push_back({ "Gradient descent with three parameters",
        [](float x) { 
            JanKadlec params;
            params.k0 = 1591369693;
            params.k1 = -2.13550628f;
            params.k2 = 2.43447248f;
            return quick_invsqrt_jan_kadlec(x, params);
        } 
    });

    std::vector<float> X_in;
    std::vector<float> Y_target;
    const auto push_sample = [&](float x) {
        const float y_target = 1.0f/std::sqrt(x);
        X_in.push_back(x);
        Y_target.push_back(y_target);
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
    const size_t N = X_in.size();
    printf("total_samples = %zu\n\n", N);

    std::vector<float> E_pred;
    E_pred.resize(N);
    for (const auto& benchmark: benchmarks) {
        float mean_abs_error = 0.0f;
        float mean_error = 0.0f;
        float max_error = 0.0f;
        float min_error = 0.0f;
        for (size_t i = 0; i < N; i++) {
            const float x = X_in[i];
            const float y_target = Y_target[i];
            const float y_pred = benchmark.invsqrt(x);
            const float error = (y_pred/y_target)-1.0f;
            E_pred[i] = error;
            mean_abs_error += std::abs(error);
            mean_error += error;
            min_error = (error < min_error) ? error : min_error;
            max_error = (error > max_error) ? error : max_error;
        }
        mean_abs_error /= float(N);
        mean_error /= float(N);
        float std_error = 0.0f;
        for (size_t i = 0; i < N; i++) {
            const float e = E_pred[i] - mean_error;
            std_error += (e*e);
        }
        std_error /= float(N);
        std_error = std::sqrt(std_error);

        printf("name = %s\n", benchmark.name);
        printf("mean_abs_error = %+.6f %%\n", mean_abs_error*1e2);
        printf("max_error      = %+.6f %%\n", max_error*1e2);
        printf("min_error      = %+.6f %%\n", min_error*1e2);
        printf("mean+std error = %+.6f %% +/- %.6f %%\n", mean_error*1e2, std_error*1e2);
        printf("\n");
    }

    return 0;
}

