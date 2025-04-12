/**
 * 文件名: convolution_hip_advanced.cu
 * 描述: 使用HIP实现的高级卷积计算，采用常量内存和纹理内存优化（适配ROCm/HIP平台）
 * 
 * 卷积原理:
 * 卷积是信号处理和图像处理中的基本操作。在离散情况下，它表示为：
 * (f * g)[n] = Σ f[m] * g[n-m]
 * 
 * 在图像处理中，二维卷积公式为:
 * O[i,j] = Σ Σ I[i+m,j+n] * K[m,n]
 * 
 * 高级HIP优化思路:
 * 1. 每个线程处理输出矩阵的一个元素
 * 2. 使用常量内存存储卷积核（适合小卷积核的只读访问）
 * 3. 使用纹理内存存储输入图像（利用空间局部性和硬件加速纹理缓存）
 * 4. 使用共享内存存储输入数据块，减少全局内存访问
 * 5. 分块处理大型输入矩阵，减少内存占用
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <math.h>

// HIP错误检查宏
#define HIP_CHECK_ERROR(err) \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s, at line %d\n", hipGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    }

// 常量内存用于存储卷积核 - 适合小尺寸卷积核
__constant__ float const_kernel[25];  // 最大支持5x5卷积核

// 声明纹理内存引用 - 用于存储输入图像
// 为HIP平台调整纹理类型
typedef texture<float, 2, hipTextureReadMode::hipReadModeElementType> floatTexture;
static floatTexture tex_input;

// 基本版本的卷积核函数（使用常量内存存储卷积核）
__global__ void convolutionKernelConstant(float* output, 
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
                // 从纹理内存读取输入数据，从常量内存读取卷积核
                float inputValue = tex2D(tex_input, static_cast<float>(inputCol), static_cast<float>(inputRow));
                sum += inputValue * const_kernel[ki * kernelCols + kj];
            }
        }
        
        // 写入结果
        output[row * outputCols + col] = sum;
    }
}

// 使用共享内存存储输入数据块的高级卷积核函数
__global__ void convolutionKernelSharedAdvanced(float* output, 
                                                int inputRows, int inputCols, 
                                                int kernelRows, int kernelCols,
                                                int outputRows, int outputCols) {
    // 定义共享内存块大小 (块大小 + 卷积核大小 - 1)
    extern __shared__ float sharedInput[];
    
    // 计算当前线程在块内的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 计算当前块要处理的输出区域
    int blockStartRow = blockIdx.y * blockDim.y;
    int blockStartCol = blockIdx.x * blockDim.x;
    
    // 计算当前线程负责的输出和输入位置
    int outRow = blockStartRow + ty;
    int outCol = blockStartCol + tx;
    
    // 计算共享内存中的尺寸
    int sharedWidth = blockDim.x + kernelCols - 1;
    int sharedHeight = blockDim.y + kernelRows - 1;
    
    // 加载输入数据到共享内存（包括边界数据）
    // 每个线程负责加载一个或多个元素
    for (int i = ty; i < sharedHeight; i += blockDim.y) {
        for (int j = tx; j < sharedWidth; j += blockDim.x) {
            int loadRow = blockStartRow + i - (kernelRows / 2);
            int loadCol = blockStartCol + j - (kernelCols / 2);
            
            // 边界检查
            if (loadRow >= 0 && loadRow < inputRows && loadCol >= 0 && loadCol < inputCols) {
                // 使用纹理内存读取输入数据
                sharedInput[i * sharedWidth + j] = tex2D(tex_input, static_cast<float>(loadCol), static_cast<float>(loadRow));
            } else {
                sharedInput[i * sharedWidth + j] = 0.0f;
            }
        }
    }
    
    // 确保所有线程都完成加载
    __syncthreads();
    
    // 确保在输出矩阵范围内
    if (outRow < outputRows && outCol < outputCols) {
        float sum = 0.0f;
        
        // 应用卷积核
        for (int ki = 0; ki < kernelRows; ki++) {
            for (int kj = 0; kj < kernelCols; kj++) {
                // 从共享内存读取输入数据，从常量内存读取卷积核
                int sharedRow = ty + ki;
                int sharedCol = tx + kj;
                sum += sharedInput[sharedRow * sharedWidth + sharedCol] * const_kernel[ki * kernelCols + kj];
            }
        }
        
        // 写入结果
        output[outRow * outputCols + outCol] = sum;
    }
}

// 为矩阵分配CPU内存
float* allocate_matrix_cpu(int rows, int cols) {
    float* matrix = (float*)malloc(rows * cols * sizeof(float));
    if (!matrix) {
        fprintf(stderr, "CPU memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return matrix;
}

// 为矩阵分配GPU内存
float* allocate_matrix_gpu(int rows, int cols) {
    float* dev_matrix;
    hipError_t err = hipMalloc((void**)&dev_matrix, rows * cols * sizeof(float));
    HIP_CHECK_ERROR(err);
    return dev_matrix;
}

// 创建随机矩阵
void create_random_matrix(float* matrix, int rows, int cols, float min_val, float max_val) {
    // 初始化随机数生成器
    srand((unsigned int)time(NULL));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // 生成范围在[min_val, max_val]的随机浮点数
            float random_val = min_val + ((float)rand() / RAND_MAX) * (max_val - min_val);
            matrix[i * cols + j] = random_val;
        }
    }
}

// 打印矩阵
void print_matrix(float* matrix, int rows, int cols, const char* name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// 使用CPU执行卷积计算(用于结果比较)
void convolve_cpu(float* input, float* kernel, float* output,
                 int inputRows, int inputCols,
                 int kernelRows, int kernelCols) {
    // 计算输出矩阵的尺寸
    int outputRows = inputRows - kernelRows + 1;
    int outputCols = inputCols - kernelCols + 1;
    
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
}

// 使用高级HIP功能执行卷积计算
void convolve_hip_advanced(float* h_input, float* h_kernel, float* h_output,
                          int inputRows, int inputCols,
                          int kernelRows, int kernelCols,
                          bool useSharedMemory = false) {
    // 计算输出矩阵的尺寸
    int outputRows = inputRows - kernelRows + 1;
    int outputCols = inputCols - kernelCols + 1;
    
    // 分配设备内存
    float* d_output = allocate_matrix_gpu(outputRows, outputCols);
    float* d_input = allocate_matrix_gpu(inputRows, inputCols);
    
    // 将输入数据复制到设备内存
    hipError_t err = hipMemcpy(d_input, h_input, inputRows * inputCols * sizeof(float), hipMemcpyHostToDevice);
    HIP_CHECK_ERROR(err);
    
    // 创建HIP格式描述符
    hipChannelFormatDesc channelDesc = hipCreateChannelDesc<float>();
    
    // 创建HIP数组用于纹理内存
    hipArray* hipArray = nullptr;
    err = hipMallocArray(&hipArray, &channelDesc, inputCols, inputRows);
    HIP_CHECK_ERROR(err);
    
    // 将输入数据复制到HIP数组
    err = hipMemcpyToArray(hipArray, 0, 0, h_input, inputRows * inputCols * sizeof(float), hipMemcpyHostToDevice);
    HIP_CHECK_ERROR(err);
    
    // 设置纹理参数
    hipResourceDesc resDesc = {};
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = hipResourceTypeArray;
    resDesc.res.array.array = hipArray;
    
    // 设置纹理描述符
    hipTextureDesc texDesc = {};
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = hipAddressModeClamp;
    texDesc.addressMode[1] = hipAddressModeClamp;
    texDesc.filterMode = hipFilterModePoint;
    texDesc.readMode = hipReadModeElementType;
    texDesc.normalizedCoords = 0;
    
    // 创建纹理对象
    hipTextureObject_t texObj = 0;
    err = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    HIP_CHECK_ERROR(err);
    
    // 将卷积核复制到常量内存
    err = hipMemcpyToSymbol(HIP_SYMBOL(const_kernel), h_kernel, kernelRows * kernelCols * sizeof(float));
    HIP_CHECK_ERROR(err);
    
    // 定义HIP线程块和网格大小
    dim3 blockSize(16, 16);
    dim3 gridSize((outputCols + blockSize.x - 1) / blockSize.x, 
                  (outputRows + blockSize.y - 1) / blockSize.y);
    
    // 根据参数选择使用基础版本还是共享内存优化版本
    if (useSharedMemory) {
        // 计算共享内存大小 - 用于存储输入数据块
        int sharedWidth = blockSize.x + kernelRows - 1;
        int sharedHeight = blockSize.y + kernelCols - 1;
        int sharedMemSize = sharedWidth * sharedHeight * sizeof(float);
        
        // 启动HIP核函数 - 共享内存优化版本
        hipLaunchKernelGGL(
            convolutionKernelSharedAdvanced,
            gridSize,
            blockSize,
            sharedMemSize,
            0,
            d_output,
            inputRows, inputCols,
            kernelRows, kernelCols,
            outputRows, outputCols
        );
    } else {
        // 启动HIP核函数 - 常量内存版本
        hipLaunchKernelGGL(
            convolutionKernelConstant,
            gridSize,
            blockSize,
            0,
            0,
            d_output,
            inputRows, inputCols,
            kernelRows, kernelCols,
            outputRows, outputCols
        );
    }
    
    // 检查内核执行错误
    err = hipGetLastError();
    HIP_CHECK_ERROR(err);
    
    // 同步设备，确保计算完成
    err = hipDeviceSynchronize();
    HIP_CHECK_ERROR(err);
    
    // 将结果从设备内存复制回主机内存
    err = hipMemcpy(h_output, d_output, outputRows * outputCols * sizeof(float), hipMemcpyDeviceToHost);
    HIP_CHECK_ERROR(err);
    
    // 销毁纹理对象
    err = hipDestroyTextureObject(texObj);
    HIP_CHECK_ERROR(err);
    
    // 释放设备内存
    hipFreeArray(hipArray);
    hipFree(d_input);
    hipFree(d_output);
}

// 创建边缘检测卷积核
void create_edge_detection_kernel(float* kernel) {
    // 拉普拉斯算子卷积核
    float laplacian[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    
    for (int i = 0; i < 9; i++) {
        kernel[i] = laplacian[i];
    }
}

// 创建高斯平滑卷积核
void create_gaussian_kernel(float* kernel, int size, float sigma) {
    float sum = 0.0f;
    int center = size / 2;
    
    // 计算高斯函数值
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int x = i - center;
            int y = j - center;
            // 高斯函数：G(x,y) = (1/(2*pi*sigma^2)) * e^(-(x^2+y^2)/(2*sigma^2))
            float value = exp(-(x*x + y*y) / (2 * sigma * sigma)) / (2 * 3.14159f * sigma * sigma);
            kernel[i * size + j] = value;
            sum += value;
        }
    }
    
    // 归一化卷积核（确保所有元素和为1）
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

// 计算两个矩阵之间的最大差异
float matrix_diff(float* a, float* b, int rows, int cols) {
    float max_diff = 0.0f;
    
    for (int i = 0; i < rows * cols; i++) {
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

// 打印设备属性
void print_device_properties() {
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    HIP_CHECK_ERROR(err);
    
    printf("检测到 %d 个HIP设备\n", deviceCount);
    
    if (deviceCount == 0) {
        printf("没有可用的HIP设备! 程序将退出。\n");
        exit(EXIT_FAILURE);
    }
    
    hipDeviceProp_t deviceProp;
    for (int i = 0; i < deviceCount; i++) {
        err = hipGetDeviceProperties(&deviceProp, i);
        HIP_CHECK_ERROR(err);
        
        printf("设备 %d: %s\n", i, deviceProp.name);
        printf("  计算能力: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  多处理器数量: %d\n", deviceProp.multiProcessorCount);
        printf("  全局内存: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  最大线程数/块: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  最大线程数/多处理器: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  时钟频率: %.2f GHz\n", deviceProp.clockRate / 1000000.0f);
        printf("  共享内存/块: %d KB\n", deviceProp.sharedMemPerBlock / 1024);
        printf("  寄存器/块: %d\n", deviceProp.regsPerBlock);
        printf("\n");
    }
}

int main() {
    printf("=== 高级HIP卷积计算程序 ===\n\n");
    
    // 打印HIP设备信息
    print_device_properties();
    
    // 设置矩阵尺寸
    int inputRows = 8, inputCols = 8;
    int kernelRows = 3, kernelCols = 3;
    int outputRows = inputRows - kernelRows + 1;
    int outputCols = inputCols - kernelCols + 1;
    
    // 分配主机内存
    float* h_input = allocate_matrix_cpu(inputRows, inputCols);
    float* h_kernel = allocate_matrix_cpu(kernelRows, kernelCols);
    float* h_output_cpu = allocate_matrix_cpu(outputRows, outputCols);
    float* h_output_constant = allocate_matrix_cpu(outputRows, outputCols);
    float* h_output_shared = allocate_matrix_cpu(outputRows, outputCols);
    
    // 生成随机输入矩阵
    create_random_matrix(h_input, inputRows, inputCols, 0.0f, 10.0f);
    
    // 创建边缘检测卷积核
    create_edge_detection_kernel(h_kernel);
    
    // 打印输入矩阵和卷积核
    print_matrix(h_input, inputRows, inputCols, "输入矩阵");
    print_matrix(h_kernel, kernelRows, kernelCols, "卷积核 (拉普拉斯算子)");
    
    // === CPU卷积计算 ===
    clock_t cpu_start = clock();
    convolve_cpu(h_input, h_kernel, h_output_cpu, inputRows, inputCols, kernelRows, kernelCols);
    clock_t cpu_end = clock();
    double cpu_time = calculate_execution_time(cpu_start, cpu_end);
    
    // === HIP卷积计算 (常量内存版本) ===
    clock_t hip_constant_start = clock();
    convolve_hip_advanced(h_input, h_kernel, h_output_constant, inputRows, inputCols, kernelRows, kernelCols, false);
    clock_t hip_constant_end = clock();
    double hip_constant_time = calculate_execution_time(hip_constant_start, hip_constant_end);
    
    // === HIP卷积计算 (共享内存优化版本) ===
    clock_t hip_shared_start = clock();
    convolve_hip_advanced(h_input, h_kernel, h_output_shared, inputRows, inputCols, kernelRows, kernelCols, true);
    clock_t hip_shared_end = clock();
    double hip_shared_time = calculate_execution_time(hip_shared_start, hip_shared_end);
    
    // 打印结果
    print_matrix(h_output_cpu, outputRows, outputCols, "CPU卷积结果");
    print_matrix(h_output_constant, outputRows, outputCols, "HIP卷积结果 (常量内存版本)");
    print_matrix(h_output_shared, outputRows, outputCols, "HIP卷积结果 (共享内存版本)");
    
    // 验证结果
    float diff_cpu_constant = matrix_diff(h_output_cpu, h_output_constant, outputRows, outputCols);
    float diff_cpu_shared = matrix_diff(h_output_cpu, h_output_shared, outputRows, outputCols);
    
    printf("CPU与HIP常量内存版本的最大差异: %f\n", diff_cpu_constant);
    printf("CPU与HIP共享内存版本的最大差异: %f\n", diff_cpu_shared);
    
    // 输出性能比较
    printf("\n=== 性能比较 ===\n");
    printf("CPU执行时间: %.2f 毫秒\n", cpu_time);
    printf("HIP常量内存版本执行时间: %.2f 毫秒 (加速比: %.2fx)\n", 
           hip_constant_time, cpu_time / hip_constant_time);
    printf("HIP共享内存版本执行时间: %.2f 毫秒 (加速比: %.2fx)\n", 
           hip_shared_time, cpu_time / hip_shared_time);
    
    // 尝试测试更大规模的矩阵
    printf("\n=== 大规模矩阵测试 ===\n");
    
    // 释放之前的内存
    free(h_input);
    free(h_kernel);
    free(h_output_cpu);
    free(h_output_constant);
    free(h_output_shared);
    
    // 更大的测试矩阵
    inputRows = 1024;
    inputCols = 1024;
    kernelRows = 5;
    kernelCols = 5;
    outputRows = inputRows - kernelRows + 1;
    outputCols = inputCols - kernelCols + 1;
    
    // 重新分配内存
    h_input = allocate_matrix_cpu(inputRows, inputCols);
    h_kernel = allocate_matrix_cpu(kernelRows, kernelCols);
    h_output_cpu = allocate_matrix_cpu(outputRows, outputCols);
    h_output_constant = allocate_matrix_cpu(outputRows, outputCols);
    h_output_shared = allocate_matrix_cpu(outputRows, outputCols);
    
    // 生成随机大矩阵和高斯卷积核
    create_random_matrix(h_input, inputRows, inputCols, 0.0f, 1.0f);
    create_gaussian_kernel(h_kernel, kernelRows, 1.0f);
    
    printf("大规模测试: 输入矩阵 %dx%d, 卷积核 %dx%d\n", inputRows, inputCols, kernelRows, kernelCols);
    
    // === CPU卷积计算 (大规模) ===
    cpu_start = clock();
    convolve_cpu(h_input, h_kernel, h_output_cpu, inputRows, inputCols, kernelRows, kernelCols);
    cpu_end = clock();
    cpu_time = calculate_execution_time(cpu_start, cpu_end);
    
    // === HIP卷积计算 (常量内存版本, 大规模) ===
    hip_constant_start = clock();
    convolve_hip_advanced(h_input, h_kernel, h_output_constant, inputRows, inputCols, kernelRows, kernelCols, false);
    hip_constant_end = clock();
    hip_constant_time = calculate_execution_time(hip_constant_start, hip_constant_end);
    
    // === HIP卷积计算 (共享内存优化版本, 大规模) ===
    hip_shared_start = clock();
    convolve_hip_advanced(h_input, h_kernel, h_output_shared, inputRows, inputCols, kernelRows, kernelCols, true);
    hip_shared_end = clock();
    hip_shared_time = calculate_execution_time(hip_shared_start, hip_shared_end);
    
    // 验证结果
    diff_cpu_constant = matrix_diff(h_output_cpu, h_output_constant, outputRows, outputCols);
    diff_cpu_shared = matrix_diff(h_output_cpu, h_output_shared, outputRows, outputCols);
    
    printf("大规模测试 - CPU与HIP常量内存版本的最大差异: %f\n", diff_cpu_constant);
    printf("大规模测试 - CPU与HIP共享内存版本的最大差异: %f\n", diff_cpu_shared);
    
    // 输出大规模测试性能比较
    printf("\n=== 大规模测试性能比较 ===\n");
    printf("CPU执行时间: %.2f 毫秒\n", cpu_time);
    printf("HIP常量内存版本执行时间: %.2f 毫秒 (加速比: %.2fx)\n", 
           hip_constant_time, cpu_time / hip_constant_time);
    printf("HIP共享内存版本执行时间: %.2f 毫秒 (加速比: %.2fx)\n", 
           hip_shared_time, cpu_time / hip_shared_time);
    
    // 释放内存
    free(h_input);
    free(h_kernel);
    free(h_output_cpu);
    free(h_output_constant);
    free(h_output_shared);
    
    // 重置HIP设备
    hipDeviceReset();
    
    printf("\n程序执行完毕\n");
    
    return 0;
} 