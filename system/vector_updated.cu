#include <iostream>
#include <cuda.h>
#include <chrono>

#define N 10000000  

using namespace std;

// GPU kernel to add two vectors
__global__ void vectorAdd(float *A, float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void vectorAddCPU(const float *A, const float *B, float *C) {
    for (int i = 0; i < N; ++i)
        C[i] = A[i] + B[i];
}

int main() {
    float *A = new float[N];
    float *B = new float[N];
    float *C_cpu = new float[N];
    float *C_gpu = new float[N];

    for (int i = 0; i < N; ++i) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    auto start_cpu = chrono::high_resolution_clock::now();
    vectorAddCPU(A, B, C_cpu);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double> cpu_time=end_cpu - start_cpu;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    auto start_gpu = chrono::high_resolution_clock::now();
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_time=end_gpu - start_gpu;

    cudaMemcpy(C_gpu, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Sample Results (First 5 Elements):\n";
    for (int i = 0; i < 5; ++i)
        cout << "A[" << i << "] + B[" << i << "] = " << C_gpu[i] << "\n";

    cout << "\nExecution Time:\n";
    cout << "CPU: " << cpu_time.count() << " s\n";
    cout << "GPU: " << gpu_time.count() << " s\n";
    cout << "Speedup Factor (CPU/GPU): " << cpu_time.count() / gpu_time.count() << "\n";

    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] A; delete[] B; delete[] C_cpu; delete[] C_gpu;

    return 0;
}
