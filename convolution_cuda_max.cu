/**
 * 文件名: convolution_cuda_max.cu
 * 描述: 使用CUDA实现的卷积计算 (针对 512x512 矩阵)
 * 
 * 卷积原理:
 * 卷积是信号处理和图像处理中的基本操作。在离散情况下，它表示为：
 * (f * g)[n] = Σ f[m] * g[n-m]
 * 
 * 在图像处理中，二维卷积公式为:
 * O[i,j] = Σ Σ I[i+m,j+n] * K[m,n]
 * 
 * CUDA并行化思路:
 * 1. 每个CUDA线程负责计算输出矩阵的一个元素
 * 2. 利用GPU的大规模并行计算能力加速卷积运算
 * 3. 对共享内存和全局内存访问进行优化
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <iomanip>

// CUDA错误检查宏
#define CUDA_CHECK_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s, at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    }

// 设备端卷积计算核函数 - 基础版本
__global__ void convolutionKernel(float* input, float* kernel, float* output, 
                                  int inputRows, int inputCols, 
                                  int kernelRows, int kernelCols,
                                  int outputRows, int outputCols) {
    // 计算当前线程负责的输出矩阵位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 确保在输出矩阵范围内
    if (row < outputRows && col < outputCols) {
        float sum = 0.0f;
        
        // 应用卷积核
        for (int ki = 0; ki < kernelRows; ki++) {
            for (int kj = 0; kj < kernelCols; kj++) {
                int inputRow = row + ki;
                int inputCol = col + kj;
                sum += input[inputRow * inputCols + inputCol] * kernel[ki * kernelCols + kj];
            }
        }
        
        // 写入结果
        output[row * outputCols + col] = sum;
    }
}

// 设备端卷积计算核函数 - 使用共享内存的优化版本 (注意: 对512x512内核不适用)
__global__ void convolutionKernelShared(float* input, float* kernel, float* output, 
                                        int inputRows, int inputCols, 
                                        int kernelRows, int kernelCols,
                                        int outputRows, int outputCols) {
    // 此共享内存实现假设内核可以完全放入共享内存，对于512x512内核通常不可行
    // 需要根据实际共享内存大小进行调整或使用不同的优化策略
    extern __shared__ float sharedKernel[]; 
    
    // 计算当前线程负责的输出矩阵位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 示例：将卷积核加载到共享内存中 - 需要修改以适应大内核和有限的共享内存
    // 以下加载方式仅适用于内核小于块大小的情况
    if (threadIdx.y < kernelRows && threadIdx.x < kernelCols) {
         // 实际加载需要更复杂的逻辑，例如分块加载
        // sharedKernel[threadIdx.y * kernelCols + threadIdx.x] = kernel[threadIdx.y * kernelCols + threadIdx.x];
    }
    
    // 确保所有线程都加载完成
    __syncthreads();
    
    // 确保在输出矩阵范围内
    if (row < outputRows && col < outputCols) {
        float sum = 0.0f;
        
        // 应用卷积核 - 从共享内存读取卷积核数据 (需要确保已正确加载)
        for (int ki = 0; ki < kernelRows; ki++) {
            for (int kj = 0; kj < kernelCols; kj++) {
                int inputRow = row + ki;
                int inputCol = col + kj;
                 // sum += input[inputRow * inputCols + inputCol] * sharedKernel[ki * kernelCols + kj]; // 需要确保 sharedKernel 已加载
                 // 回退到全局内存读取作为示例
                 sum += input[inputRow * inputCols + inputCol] * kernel[ki * kernelCols + kj];
            }
        }
        
        // 写入结果
        output[row * outputCols + col] = sum;
    }
}

// 为矩阵分配CPU内存
float* allocate_matrix_cpu(int rows, int cols) {
    float* matrix = (float*)malloc(rows * cols * sizeof(float));
    if (!matrix) {
        fprintf(stderr, "CPU memory allocation failed for %dx%d matrix\n", rows, cols);
        exit(EXIT_FAILURE);
    }
    printf("Allocated CPU memory for %dx%d matrix\n", rows, cols);
    return matrix;
}

// 为矩阵分配GPU内存
float* allocate_matrix_gpu(int rows, int cols) {
    float* dev_matrix;
    cudaError_t err = cudaMalloc((void**)&dev_matrix, rows * cols * sizeof(float));
    CUDA_CHECK_ERROR(err);
    printf("Allocated GPU memory for %dx%d matrix\n", rows, cols);
    return dev_matrix;
}

// 创建随机矩阵
void create_random_matrix(float* matrix, int rows, int cols, float min_val, float max_val) {
    // 初始化随机数生成器
    // 注意：如果在短时间内多次调用，使用 time(NULL) 可能导致种子相同
    // 可以考虑更可靠的种子生成方式或只初始化一次
    srand((unsigned int)time(NULL)); 
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // 生成范围在[min_val, max_val]的随机浮点数
            float random_val = min_val + ((float)rand() / RAND_MAX) * (max_val - min_val);
            matrix[i * cols + j] = random_val;
        }
    }
    printf("Generated random data for %dx%d matrix\n", rows, cols);
}

// 打印矩阵 (只打印部分元素以避免过大输出)
void print_matrix_partial(float* matrix, int rows, int cols, const char* name, int print_rows = 5, int print_cols = 5) {
    printf("%s (%dx%d) - Showing first %dx%d elements:\n", name, rows, cols, print_rows, print_cols);
    
    for (int i = 0; i < std::min(rows, print_rows); i++) {
        for (int j = 0; j < std::min(cols, print_cols); j++) {
            printf("%8.4f ", matrix[i * cols + j]);
        }
        if (cols > print_cols) printf("...");
        printf("\n");
    }
    if (rows > print_rows) printf("...\n");
    printf("\n");
}

// 使用CPU执行卷积计算(用于结果比较 - 对大矩阵可能非常慢)
void convolve_cpu(float* input, float* kernel, float* output,
                 int inputRows, int inputCols,
                 int kernelRows, int kernelCols) {
    // 计算输出矩阵的尺寸
    int outputRows = inputRows - kernelRows + 1;
    int outputCols = inputCols - kernelCols + 1;
    
    printf("Starting CPU convolution for %dx%d output...\n", outputRows, outputCols);
    // 计算卷积
    for (int i = 0; i < outputRows; i++) {
        for (int j = 0; j < outputCols; j++) {
            // 对当前位置应用卷积核
            float sum = 0.0f;
            for (int ki = 0; ki < kernelRows; ki++) {
                for (int kj = 0; kj < kernelCols; kj++) {
                    sum += input[(i + ki) * inputCols + (j + kj)] * kernel[ki * kernelCols + kj];
                }
            }
            output[i * outputCols + j] = sum;
        }
    }
    printf("CPU convolution finished.\n");
}

// 使用CUDA执行卷积计算
void convolve_cuda(float* h_input, float* h_kernel, float* h_output,
                  int inputRows, int inputCols,
                  int kernelRows, int kernelCols,
                  bool useSharedMemory = false) // 注意: shared memory 对大内核无效
                  {
    // 计算输出矩阵的尺寸
    int outputRows = inputRows - kernelRows + 1;
    int outputCols = inputCols - kernelCols + 1;
    
    printf("CUDA convolution setup: Output %dx%d\n", outputRows, outputCols);
    
    // 分配设备内存
    float* d_input = allocate_matrix_gpu(inputRows, inputCols);
    float* d_kernel = allocate_matrix_gpu(kernelRows, kernelCols);
    float* d_output = allocate_matrix_gpu(outputRows, outputCols);
    
    // 将输入数据从主机内存复制到设备内存
    printf("Copying data Host -> Device...\n");
    cudaError_t err = cudaMemcpy(d_input, h_input, inputRows * inputCols * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(err);
    
    err = cudaMemcpy(d_kernel, h_kernel, kernelRows * kernelCols * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(err);
    printf("Data copy finished.\n");
    
    // 定义CUDA线程块和网格大小
    dim3 blockSize(16, 16); // 常用的块大小
    // 对于 1x1 输出，只需要一个块
    dim3 gridSize((outputCols + blockSize.x - 1) / blockSize.x, 
                  (outputRows + blockSize.y - 1) / blockSize.y);
    printf("CUDA kernel launch parameters: Grid=(%d,%d), Block=(%d,%d)\n", gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    // 根据参数选择使用基础版本还是共享内存优化版本
    if (useSharedMemory) {
        // 计算共享内存大小 - 用于存储卷积核
        int sharedMemSize = kernelRows * kernelCols * sizeof(float);
        printf("Attempting Shared Memory Kernel (size: %d bytes)... **NOTE: Likely unsuitable for large kernels!**\n", sharedMemSize);
        
        // 启动CUDA核函数 - 共享内存版本
        convolutionKernelShared<<<gridSize, blockSize, sharedMemSize>>>(
            d_input, d_kernel, d_output,
            inputRows, inputCols,
            kernelRows, kernelCols,
            outputRows, outputCols
        );
    } else {
        printf("Launching Basic Kernel...\n");
        // 启动CUDA核函数 - 基础版本
        convolutionKernel<<<gridSize, blockSize>>>(
            d_input, d_kernel, d_output,
            inputRows, inputCols,
            kernelRows, kernelCols,
            outputRows, outputCols
        );
    }
    
    // 检查内核执行错误
    err = cudaGetLastError();
    CUDA_CHECK_ERROR(err); // 会捕获异步错误
    printf("Kernel launch submitted. Synchronizing device...\n");
    
    // 同步设备，确保计算完成
    err = cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(err);
    printf("Device synchronization complete. Kernel finished.\n");
    
    // 将结果从设备内存复制回主机内存
    printf("Copying result Device -> Host...\n");
    err = cudaMemcpy(h_output, d_output, outputRows * outputCols * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(err);
    printf("Result copy finished.\n");
    
    // 释放设备内存
    printf("Freeing GPU memory...\n");
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    printf("GPU memory freed.\n");
}

// 计算两个矩阵之间的最大差异 (仅适用于相同尺寸)
float matrix_diff(float* a, float* b, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return 0.0f; // Handle 1x1 case
    float max_diff = 0.0f;
    int num_elements = rows * cols;
    
    for (int i = 0; i < num_elements; i++) {
        float diff = fabs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    
    return max_diff;
}

// 计算执行时间 (毫秒)
double calculate_execution_time(clock_t start, clock_t end) {
    return ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
}

int main() {
    // 设置控制台代码页为简体中文GBK，以支持中文显示
    #ifdef _WIN32
    system("chcp 936 > nul");
    #endif
    
    printf("=== CUDA Convolution Calculation Program (512x512 Test) ===\n\n");
    
    // 输出CUDA设备信息
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    CUDA_CHECK_ERROR(err);
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-enabled devices found! Exiting.\n");
        return EXIT_FAILURE;
    }
    
    printf("Detected %d CUDA device(s)\n", deviceCount);
    
    cudaDeviceProp deviceProp;
    // 使用设备0
    int deviceId = 0; 
    err = cudaGetDeviceProperties(&deviceProp, deviceId);
    CUDA_CHECK_ERROR(err);
    printf("Using Device %d: %s\n", deviceId, deviceProp.name);
    printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Total Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared Memory Per Block: %zu bytes\n", deviceProp.sharedMemPerBlock);
    printf("  Max Threads Per Block: %d\n\n", deviceProp.maxThreadsPerBlock);
    
    // 设置矩阵尺寸
    int inputRows = 512, inputCols = 512;
    int kernelRows = 512, kernelCols = 512;
    // 计算输出尺寸
    int outputRows = inputRows - kernelRows + 1; // 512 - 512 + 1 = 1
    int outputCols = inputCols - kernelCols + 1; // 512 - 512 + 1 = 1
    
    printf("--- Configuration ---\n");
    printf("Input Matrix Size: %d x %d\n", inputRows, inputCols);
    printf("Kernel Matrix Size: %d x %d\n", kernelRows, kernelCols);
    printf("Output Matrix Size: %d x %d\n\n", outputRows, outputCols);
    
    // 分配主机内存
    printf("--- Allocating Host Memory ---\n");
    float* h_input = allocate_matrix_cpu(inputRows, inputCols);
    float* h_kernel = allocate_matrix_cpu(kernelRows, kernelCols);
    float* h_output_cuda = allocate_matrix_cpu(outputRows, outputCols); // 1x1 output
    // float* h_output_cpu = allocate_matrix_cpu(outputRows, outputCols); // Optional: For CPU comparison (VERY SLOW!)
    
    // 生成随机输入矩阵和卷积核
    printf("\n--- Generating Random Matrices ---\n");
    create_random_matrix(h_input, inputRows, inputCols, 0.0f, 1.0f);
    create_random_matrix(h_kernel, kernelRows, kernelCols, -0.5f, 0.5f);
    
    // 打印部分输入矩阵和卷积核 (可选)
    // print_matrix_partial(h_input, inputRows, inputCols, "Input Matrix");
    // print_matrix_partial(h_kernel, kernelRows, kernelCols, "Kernel Matrix");
    
    // === CPU卷积计算 (可选, 极慢) ===
    // printf("\n--- CPU Convolution (Optional, may take a long time) ---\n");
    // clock_t cpu_start = clock();
    // convolve_cpu(h_input, h_kernel, h_output_cpu, inputRows, inputCols, kernelRows, kernelCols);
    // clock_t cpu_end = clock();
    // double cpu_time = calculate_execution_time(cpu_start, cpu_end);
    // printf("CPU Convolution Result: %f\n", h_output_cpu[0]);
    // printf("CPU Execution Time: %.2f ms\n", cpu_time);
    
    // === CUDA卷积计算 (基础版本) ===
    printf("\n--- CUDA Convolution (Basic Kernel) ---\n");
    clock_t cuda_start = clock();
    // 使用基础内核 (useSharedMemory = false)
    convolve_cuda(h_input, h_kernel, h_output_cuda, inputRows, inputCols, kernelRows, kernelCols, false);
    clock_t cuda_end = clock();
    double cuda_time = calculate_execution_time(cuda_start, cuda_end);
    
    // 打印结果
    printf("\n--- Results ---\n");
    printf("CUDA Convolution Result (Basic): %f\n", h_output_cuda[0]);
    
    // 输出性能
    printf("CUDA Basic Kernel Execution Time: %.2f ms\n", cuda_time);
    
    // (可选) 比较 CPU 和 CUDA 结果
    // float diff_cpu_cuda = matrix_diff(h_output_cpu, h_output_cuda, outputRows, outputCols);
    // printf("Max Difference between CPU and CUDA Basic: %f\n", diff_cpu_cuda);
    
    // 释放主机内存
    printf("\n--- Freeing Host Memory ---\n");
    free(h_input); printf("Freed h_input\n");
    free(h_kernel); printf("Freed h_kernel\n");
    free(h_output_cuda); printf("Freed h_output_cuda\n");
    // free(h_output_cpu); // if allocated
    printf("Host memory freed.\n");
    
    // 重置CUDA设备
    cudaDeviceReset();
    printf("CUDA device reset.\n");
    
    printf("\nProgram finished successfully.\n");
    
    return 0;
} 