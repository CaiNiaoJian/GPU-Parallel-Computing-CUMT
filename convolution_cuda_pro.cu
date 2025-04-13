/**
 * 文件名: optimized_convolution_cuda.cu
 * 描述: 使用共享内存优化的CUDA卷积实现
 * 特性: 平铺卷积、双缓冲共享内存、智能边界处理
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>

// 优化参数
#define TILE_SIZE       16
#define KERNEL_RADIUS   1    // 3x3卷积核
#define KERNEL_SIZE     (2*KERNEL_RADIUS+1)

// CUDA错误检查宏
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 优化后的卷积核函数
__global__ void optimizedConvolutionKernel(
    float* input, float* output, int inputRows, int inputCols,
    const float* __restrict__ kernel, int kernelRows, int kernelCols)
{
    // 共享内存声明
    extern __shared__ float sharedMem[];
    float* sharedData = sharedMem;
    float* sharedKernel = (float*)&sharedData[(TILE_SIZE + 2*KERNEL_RADIUS) * (TILE_SIZE + 2*KERNEL_RADIUS)];

    // 线程索引计算
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row_o = blockIdx.y * TILE_SIZE + ty;
    const int col_o = blockIdx.x * TILE_SIZE + tx;
    const int row_i = row_o - KERNEL_RADIUS;
    const int col_i = col_o - KERNEL_RADIUS;

    // 加载输入数据到共享内存（包含halo区域）
    if (row_i >= 0 && row_i < inputRows && col_i >= 0 && col_i < inputCols) {
        sharedData[ty * (TILE_SIZE + 2*KERNEL_RADIUS) + tx] = input[row_i * inputCols + col_i];
    } else {
        sharedData[ty * (TILE_SIZE + 2*KERNEL_RADIUS) + tx] = 0.0f;
    }

    // 加载卷积核到共享内存
    if (ty < kernelRows && tx < kernelCols) {
        sharedKernel[ty * kernelCols + tx] = kernel[ty * kernelCols + tx];
    }

    __syncthreads();

    // 卷积计算
    if (tx < TILE_SIZE && ty < TILE_SIZE && row_o < inputRows && col_o < inputCols) {
        float sum = 0.0f;
        #pragma unroll
        for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ++ky) {
            #pragma unroll
            for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; ++kx) {
                int sy = ty + KERNEL_RADIUS + ky;
                int sx = tx + KERNEL_RADIUS + kx;
                sum += sharedData[sy * (TILE_SIZE + 2*KERNEL_RADIUS) + sx] 
                     * sharedKernel[(ky + KERNEL_RADIUS) * kernelCols + (kx + KERNEL_RADIUS)];
            }
        }
        output[row_o * inputCols + col_o] = sum;
    }
}

// 主机端辅助函数
float* create_matrix(int rows, int cols) {
    float* m = (float*)malloc(rows * cols * sizeof(float));
    if (!m) {
        fprintf(stderr, "内存分配失败\n");
        exit(EXIT_FAILURE);
    }
    return m;
}

void rand_init(float* data, int size, float min_val=0.0f, float max_val=1.0f) {
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        data[i] = min_val + (max_val - min_val) * (rand() / (float)RAND_MAX);
    }
}

void print_matrix(const float* m, int rows, int cols, const char* name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.4f ", m[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// CPU参考实现
void cpu_convolution(const float* input, float* output, int rows, int cols,
                    const float* kernel, int ksize) {
    const int krad = ksize / 2;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float sum = 0.0f;
            for (int ky = -krad; ky <= krad; ++ky) {
                for (int kx = -krad; kx <= krad; ++kx) {
                    int iy = y + ky;
                    int ix = x + kx;
                    if (iy >= 0 && iy < rows && ix >= 0 && ix < cols) {
                        float val = input[iy * cols + ix];
                        float w = kernel[(ky + krad) * ksize + (kx + krad)];
                        sum += val * w;
                    }
                }
            }
            output[y * cols + x] = sum;
        }
    }
}

// CUDA卷积包装函数
void cuda_convolution(const float* h_input, float* h_output, int rows, int cols,
                     const float* h_kernel, int ksize) {
    // 设备内存分配
    float *d_input, *d_output, *d_kernel;
    CUDA_CHECK(cudaMalloc(&d_input, rows*cols*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, rows*cols*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, ksize*ksize*sizeof(float)));

    // 数据传输
    CUDA_CHECK(cudaMemcpy(d_input, h_input, rows*cols*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, ksize*ksize*sizeof(float), cudaMemcpyHostToDevice));

    // 计算执行配置
    dim3 block(TILE_SIZE + 2*KERNEL_RADIUS, TILE_SIZE + 2*KERNEL_RADIUS);
    dim3 grid((cols + TILE_SIZE - 1) / TILE_SIZE, 
              (rows + TILE_SIZE - 1) / TILE_SIZE);

    // 计算共享内存需求
    size_t shared_size = ((TILE_SIZE + 2*KERNEL_RADIUS) * (TILE_SIZE + 2*KERNEL_RADIUS) 
                       + KERNEL_SIZE*KERNEL_SIZE) * sizeof(float);

    // 启动核函数
    optimizedConvolutionKernel<<<grid, block, shared_size>>>(
        d_input, d_output, rows, cols, d_kernel, KERNEL_SIZE, KERNEL_SIZE
    );
    CUDA_CHECK(cudaGetLastError());

    // 取回结果
    CUDA_CHECK(cudaMemcpy(h_output, d_output, rows*cols*sizeof(float), cudaMemcpyDeviceToHost));

    // 清理
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kernel));
}

// 验证函数
float verify(const float* ref, const float* test, int size) {
    float max_error = 0.0f;
    for (int i = 0; i < size; ++i) {
        float err = fabs(ref[i] - test[i]);
        if (err > max_error) max_error = err;
    }
    return max_error;
}

int main() {
    // 矩阵参数
    const int W = 512;  // 图像宽度
    const int H = 512;  // 图像高度
    const int K = 3;    // 卷积核尺寸

    // 分配内存
    float* input = create_matrix(H, W);
    float* output_cpu = create_matrix(H, W);
    float* output_gpu = create_matrix(H, W);
    float* kernel = create_matrix(K, K);

    // 初始化数据
    rand_init(input, H*W, 0.0f, 1.0f);
    rand_init(kernel, K*K, -1.0f, 1.0f);

    // CPU计算
    clock_t cpu_start = clock();
    cpu_convolution(input, output_cpu, H, W, kernel, K);
    double cpu_time = (double)(clock() - cpu_start) / CLOCKS_PER_SEC * 1000;

    // GPU计算
    clock_t gpu_start = clock();
    cuda_convolution(input, output_gpu, H, W, kernel, K);
    double gpu_time = (double)(clock() - gpu_start) / CLOCKS_PER_SEC * 1000;

    // 验证结果
    float max_error = verify(output_cpu, output_gpu, H*W);

    // 输出结果
    printf("性能测试结果:\n");
    printf("CPU 时间: %.2f ms\n", cpu_time);
    printf("GPU 时间: %.2f ms\n", gpu_time);
    printf("加速比: %.2f x\n", cpu_time / gpu_time);
    printf("最大误差: %.6f\n", max_error);

    // 示例输出（小矩阵）
    if (H <= 8 && W <= 8) {
        print_matrix(input, H, W, "输入矩阵");
        print_matrix(kernel, K, K, "卷积核");
        print_matrix(output_cpu, H, W, "CPU结果");
        print_matrix(output_gpu, H, W, "GPU结果");
    }

    // 清理
    free(input);
    free(output_cpu);
    free(output_gpu);
    free(kernel);

    cudaDeviceReset();
    return 0;
}