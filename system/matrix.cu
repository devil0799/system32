#include <iostream>
#include <cuda.h>
#include <chrono>

#define N 512  

using namespace std;

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < width; i++) {
            sum += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = sum;
    }
}

// CPU matrix multiplication
void matrixMulCPU(float* A, float* B, float* C, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int i = 0; i < width; i++) {
                sum += A[row * width + i] * B[i * width + col];
            }
            C[row * width + col] = sum;
        }
    }
}

int main() {
    int size = N * N * sizeof(float);

    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C_cpu = new float[N * N];
    float* C_gpu = new float[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Time CPU matrix multiplication
    auto start_cpu = chrono::high_resolution_clock::now();
    matrixMulCPU(A, B, C_cpu, N);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double> cpu_time = end_cpu - start_cpu;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy input matrices to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    auto start_gpu = chrono::high_resolution_clock::now();
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // Copy result back to host
    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // Print first 5 results for verification
    cout << "Sample Output (First 5 Elements):\n";
    for (int i = 0; i < 5; i++) {
        cout << "C_cpu[" << i << "] = " << C_cpu[i]
             << ", C_gpu[" << i << "] = " << C_gpu[i] << "\n";
    }

    // Print execution time
    cout << "\nExecution Time:\n";
    cout << "CPU: " << cpu_time.count() << " s\n";
    cout << "GPU: " << gpu_time.count() << " s\n";
    cout << "Speedup (CPU/GPU): " << cpu_time.count() / gpu_time.count() << "\n";

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu;

    return 0;
}
