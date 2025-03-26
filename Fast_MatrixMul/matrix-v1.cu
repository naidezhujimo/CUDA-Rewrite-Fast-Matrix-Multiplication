#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>
#include <chrono>

#define N 1024         // 矩阵尺寸 N x N
#define BLOCK_SIZE 32  // 手动核函数的块大小
#define EPSILON 1e-3   // 验证误差阈值

// CUDA错误检查宏
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::cerr << "CUDA Error at line " << __LINE__ << ": "                 \
                  << cudaGetErrorString(status) << std::endl;                  \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

// cuBLAS错误检查宏
#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        std::cerr << "cuBLAS Error at line " << __LINE__ << ": "               \
                  << status << std::endl;                                      \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

// 手动优化的分块矩阵乘法核函数（行优先存储）
__global__ void matrixMulSharedKernel(float *A, float *B, float *C, int size) {
    // 动态申请共享内存
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // 计算当前线程处理的C矩阵的行和列
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // 分块循环：共需要 size/BLOCK_SIZE 次迭代
    for (int m = 0; m < (size + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        // 协作加载A和B的块到共享内存
        if (row < size && (m * BLOCK_SIZE + tx) < size){
            s_A[ty][tx] = A[row * size + (m * BLOCK_SIZE + tx)];
        }
        else{
            s_A[ty][tx] = 0.0f;
        }
        if (col < size && (m * BLOCK_SIZE + ty) < size){
            s_B[ty][tx] = B[(m * BLOCK_SIZE + ty) * size + col];
        }
        else{
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads(); // 确保块加载完成

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads(); // 确保计算完成当前块再计算下一个块
    }

    // 写入结果到C矩阵
    if (row < size && col < size) {
        C[row * size + col] = sum;
    }
}

// CPU矩阵乘法
void matrixMulCPU(float *A, float *B, float *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++){
            float sum = 0.0f;
            for (int k = 0; k < size; k++){
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

// 验证结果正确性
bool verifyResult(float *ref, float *res, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(ref[i] - res[i]) > EPSILON) {
            std::cerr << "验证失败!索引 " << i 
                      << ": ref = " << ref[i] 
                      << ", res = " << res[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // 分配主机内存
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C_cpu = new float[N * N];
    float *h_C_gpu = new float[N * N];
    float *h_C_cublas = new float[N * N];

    // 初始化矩阵A和B
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand() / RAND_MAX);
        h_B[i] = static_cast<float>(rand() / RAND_MAX);
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, N * N * sizeof(float)));

    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 预热
    printf("Warming up...\n");
    for (int i = 0; i < 3; i++) {
        matrixMulCPU(h_A, h_B, h_C_cpu, N);
        matrixMulSharedKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize(); // 确保GPU上操作完成
    }

    // 运行CPU基准测试
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_duration.count() << "s" << std::endl;

    // 运行核函数
    cudaEvent_t start_gpu, end_gpu;
    CHECK_CUDA(cudaEventCreate(&start_gpu));
    CHECK_CUDA(cudaEventCreate(&end_gpu));

    CHECK_CUDA(cudaEventRecord(start_gpu));
    matrixMulSharedKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(end_gpu));
    CHECK_CUDA(cudaEventSynchronize(end_gpu));

    float gpu_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_time_ms, start_gpu, end_gpu));
    std::cout << "Manual GPU function time: " << gpu_time_ms << "ms" << std::endl;

    // 复制结果回主机并验证
    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    if (verifyResult(h_C_cpu, h_C_gpu, N)) {
        std::cout << "The result of the manual kernel function has been verified!" << std::endl;
    }


    //运行cublas库(列优先)
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDA(cudaMemset(d_C, 0, N * N * sizeof(float))); // 清空C

    cudaEvent_t start_cublas, end_cublas;
    CHECK_CUDA(cudaEventCreate(&start_cublas));
    CHECK_CUDA(cudaEventCreate(&end_cublas));

    CHECK_CUDA(cudaEventRecord(start_cublas));
    // 计算 C = A^T * B^T 的列优先结果，等价于行优先的 C = A * B
    CHECK_CUBLAS(cublasSgemm(handle, 
        CUBLAS_OP_T,   // A^T（因为A是行优先，转置后变为列优先）
        CUBLAS_OP_T,   // B^T
        N, N, N, 
        &alpha, 
        d_A, N,       // LDA
        d_B, N, 
        &beta, 
        d_C, N));
    CHECK_CUDA(cudaEventRecord(end_cublas));
    CHECK_CUDA(cudaEventSynchronize(end_cublas));

    float cublas_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&cublas_time_ms, start_cublas, end_cublas));
    std::cout << "cuBLAS time: " << cublas_time_ms << "ms" << std::endl;

    // 复制cuBLAS结果并验证
    CHECK_CUDA(cudaMemcpy(h_C_cublas, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    if (verifyResult(h_C_cpu, h_C_cublas, N)) {
        std::cout << "The results of the cuBLAS has been verfied!" << std::endl;
    }

    // 释放资源
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;
    delete[] h_C_cublas;

    return 0;

}