#include <iostream>
#include <chrono>
#include <fstream>
#include <omp.h>

// Function for matrix multiplication on the CPU with OpenMP
void matrixMultiplyCPU(float* A, float* B, float* C, int N) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float value = 0;
            for (int k = 0; k < N; ++k) {
                value += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = value;
        }
    }
}

int main() {
    int N = 2000; // Example size of the matrix
    size_t size = N * N * sizeof(float);

    // Allocate memory for matrices
    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C = (float*)malloc(size);

    // Initialize matrices with some values
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i);
    }

    // Measure the time taken for the CPU matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    // Write the first element of the result matrix to a file
    std::ofstream outputFile("cpu_output.txt");
    if (outputFile.is_open()) {
        outputFile << C[0] << " ";
        outputFile << "\n";
        outputFile.close();
    } else {
        std::cerr << "Unable to open file for writing\n";
    }

    std::cout << "finish" << std::endl;
    std::cout << "CPU Matrix multiplication took " << duration.count() << " ms\n";

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
