cmake_minimum_required(VERSION 3.10)
project(fast_inverse_square_root)

function(add_example target)
    add_executable(${target} ${target}.cpp)
    target_compile_features(${target} PRIVATE cxx_std_17)
endfunction()

add_example(benchmark_fast_invsqrt)
add_example(derive_fast_invsqrt_quake)
add_example(derive_fast_invsqrt_jan_kadlec)
