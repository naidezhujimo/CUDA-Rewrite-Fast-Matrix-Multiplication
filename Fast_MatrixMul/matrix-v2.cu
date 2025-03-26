#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>
#include <chrono>

#define N 1024         // 矩阵尺寸 N x N
#define BLOCK_SIZE 32  // 手动核函数的块大小
#define TILE_SIZE_K 32        // K维度分块大小
#define VECTOR_SIZE 4         // 向量化加载的元素数（float4）
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
    // 双缓冲共享内存：两个Tile用于A和B
    __shared__ float s_A[2][BLOCK_SIZE][TILE_SIZE_K]; // +1解决bank冲突
    __shared__ float s_B[2][BLOCK_SIZE][BLOCK_SIZE * VECTOR_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // 每个线程块负责计算一个 BLOCK_SIZE x (BLOCK_SIZE*VECTOR_SIZE) 的C子矩阵
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE * VECTOR_SIZE + tx * VECTOR_SIZE;

    // 寄存器累加器：每个线程计算 VECTOR_SIZE x 1 的C元素
    float c[VECTOR_SIZE] = {0.0f};

    // 预加载第一个Tile到共享内存（双缓冲的前半部分）
    int load_phase = 0;
    for (int k = 0; k < size; k += TILE_SIZE_K) {
        // 协作加载A的Tile（向量化加载）
        int load_k = k + ty;
        
        if (load_k < size && row < size) {
            // 确保地址对齐
            float4 vec = reinterpret_cast<float4*>(&A[row * size + load_k * VECTOR_SIZE])[tx];
            s_A[load_phase][ty][tx * VECTOR_SIZE + 0] = vec.x;
            s_A[load_phase][ty][tx * VECTOR_SIZE + 1] = vec.y;
            s_A[load_phase][ty][tx * VECTOR_SIZE + 2] = vec.z;
            s_A[load_phase][ty][tx * VECTOR_SIZE + 3] = vec.w;
        } else {
            for (int v = 0; v < VECTOR_SIZE; ++v) {
                s_A[load_phase][ty][tx * VECTOR_SIZE + v] = 0.0f;
            }
        }

        // 协作加载B的Tile（向量化加载）
        int load_col = col + tx * VECTOR_SIZE;
        if (load_col < size && (k + ty) < size) {
            // 确保地址对齐
            float4 vec = reinterpret_cast<float4*>(&B[(k + ty) * size + load_col])[0];
            s_B[load_phase][ty][tx * VECTOR_SIZE + 0] = vec.x;
            s_B[load_phase][ty][tx * VECTOR_SIZE + 1] = vec.y;
            s_B[load_phase][ty][tx * VECTOR_SIZE + 2] = vec.z;
            s_B[load_phase][ty][tx * VECTOR_SIZE + 3] = vec.w;
        } else {
            for (int v = 0; v < VECTOR_SIZE; ++v) {
                s_B[load_phase][ty][tx * VECTOR_SIZE + v] = 0.0f;
            }
        }

        __syncthreads();

        // 循环计算所有Tiles
        for (int tk = 0; tk < TILE_SIZE_K; ++tk) {
            // 双缓冲：异步加载下一个Tile
            if (tk % 2 == 0 && k + TILE_SIZE_K < size) {
                // 加载下一个Tile到缓冲区的另一半
                int next_load_phase = load_phase ^ 1;
                int next_k = k + TILE_SIZE_K;
                int next_load_k = next_k + ty;

                if (next_load_k < size && row < size) {
                    // 确保地址对齐
                    float4 vec = reinterpret_cast<float4*>(&A[row * size + next_load_k * VECTOR_SIZE])[tx];
                    s_A[next_load_phase][ty][tx * VECTOR_SIZE + 0] = vec.x;
                    s_A[next_load_phase][ty][tx * VECTOR_SIZE + 1] = vec.y;
                    s_A[next_load_phase][ty][tx * VECTOR_SIZE + 2] = vec.z;
                    s_A[next_load_phase][ty][tx * VECTOR_SIZE + 3] = vec.w;
                } else {
                    for (int v = 0; v < VECTOR_SIZE; ++v) {
                        s_A[next_load_phase][ty][tx * VECTOR_SIZE + v] = 0.0f;
                    }
                }
                int next_load_col = col + tx * VECTOR_SIZE;
                if (next_load_col < size && (next_k + ty) < size) {
                    // 确保地址对齐
                    float4 vec = reinterpret_cast<float4*>(&B[(next_k + ty) * size + next_load_col])[0];
                    s_B[next_load_phase][ty][tx * VECTOR_SIZE + 0] = vec.x;
                    s_B[next_load_phase][ty][tx * VECTOR_SIZE + 1] = vec.y;
                    s_B[next_load_phase][ty][tx * VECTOR_SIZE + 2] = vec.z;
                    s_B[next_load_phase][ty][tx * VECTOR_SIZE + 3] = vec.w;
                } else {
                    for (int v = 0; v < VECTOR_SIZE; ++v) {
                        s_B[next_load_phase][ty][tx * VECTOR_SIZE + v] = 0.0f;
                    }
                }
                __syncthreads();
            }

            // 从共享内存读取A和B的当前Tile数据
            float a = s_A[load_phase][ty][tk];
            for (int v = 0; v < VECTOR_SIZE; ++v) {
                float b = s_B[load_phase][tk][tx * VECTOR_SIZE + v];
                c[v] += a * b;
            }
        }

        // 切换双缓冲
        load_phase ^= 1;
        __syncthreads();
    }

    // 将结果写回全局内存（向量化存储）
    if (row < size && col < size) {
        for (int v = 0; v < VECTOR_SIZE; ++v) {
            if (col + v < size) {
                C[row * size + col + v] = c[v];
            }
        }
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
    size_t pitch_A, pitch_B, pitch_C;
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMallocPitch(&d_A, &pitch_A, N * sizeof(float), N));
    CHECK_CUDA(cudaMallocPitch(&d_B, &pitch_B, N * sizeof(float), N));
    CHECK_CUDA(cudaMallocPitch(&d_C, &pitch_C, N * sizeof(float), N));

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