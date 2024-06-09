# Introduction
Explanation and derivation of fast inverse square root from Quake.
- Benchmark relative error.
- Derive single parameter in original Quake implementation
- Derive three parameters in extended form by Jan Kadlec

## Results
| Name                 |  L1-error (%)  |  L6-error (%)  |  max (%)  |  min (%)  |  avg (%)  |  std (%)  |
| -------------------- | -------------- | -------------- | --------- | --------- | --------- | --------- |
| Original Quake       |       9.51e-02 |       6.31e-16 | -6.47e-06 | -1.75e-01 | -9.51e-02 |  5.71e-02 |
| Gradient descent (1) |       9.28e-02 |       5.63e-16 |  0.00e+00 | -1.79e-01 | -9.28e-02 |  5.61e-02 |
| Jan Kadlec           |       4.21e-02 |       2.71e-18 |  6.50e-02 | -6.50e-02 |  2.02e-02 |  4.24e-02 |
| Gradient descent (3) |       3.80e-02 |       1.68e-18 |  5.82e-02 | -7.74e-02 |  1.35e-02 |  4.07e-02 |

![Results](./docs/results.png)

## Compiling code
1. Configure cmake: ```cmake . -B build --preset windows-msvc-avx2 -DCMAKE_BUILD_TYPE=Release```
2. Build applications: ```cmake --build build```
3. Run applications: ```./build/*.exe```

## Running Jupyter notebook
1. Install python.
2. Create virtual environment: ```python -m venv venv```.
3. Activate virtual environment: ```source ./venv/Scripts/activate```.
4. Install packages: ```pip install numpy matplotlib tabulate jupyter```.
5. Start notebook: ```jupyter-notebook .```.
6. Run notebook and execute cells.
