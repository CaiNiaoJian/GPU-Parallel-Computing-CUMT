#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

const int TILE_SIZE = 16;
const int KERNEL_MAX = 7;  // 支持最大7x7卷积核
const int PAD = KERNEL_MAX / 2;

// 常量内存存储卷积核
__constant__ float c_kernel[KERNEL_MAX][KERNEL_MAX];

// 基础全局内存版本
__global__ void convolution_global(
    float* input, float* output,
    int width, int height,
    int kernel_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) return;

    float sum = 0.0f;
    int k_radius = kernel_size / 2;
    
    for (int ky = -k_radius; ky <= k_radius; ++ky) {
        for (int kx = -k_radius; kx <= k_radius; ++kx) {
            int y = row + ky;
            int x = col + kx;
            if (y >= 0 && y < height && x >= 0 && x < width) {
                sum += input[y * width + x] * c_kernel[ky + k_radius][kx + k_radius];
            }
        }
    }
    output[row * width + col] = sum;
}

// 共享内存优化版本
__global__ void convolution_shared(
    float* input, float* output,
    int width, int height,
    int kernel_size)
{
    extern __shared__ float s_data[];
    
    int k_radius = kernel_size / 2;
    int tile_width = TILE_SIZE + 2 * k_radius;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int col = blockIdx.x * TILE_SIZE + tx - k_radius;
    int row = blockIdx.y * TILE_SIZE + ty - k_radius;
    
    // 加载数据到共享内存（包含halo区域）
    if (col >= 0 && col < width && row >= 0 && row < height) {
        s_data[ty * tile_width + tx] = input[row * width + col];
    } else {
        s_data[ty * tile_width + tx] = 0.0f;
    }
    
    __syncthreads();
    
    // 计算有效输出位置
    int out_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    if (out_col >= width || out_row >= height) return;
    
    // 卷积计算
    float sum = 0.0f;
    for (int ky = -k_radius; ky <= k_radius; ++ky) {
        for (int kx = -k_radius; kx <= k_radius; ++kx) {
            int sy = ty + k_radius + ky;
            int sx = tx + k_radius + kx;
            sum += s_data[sy * tile_width + sx] * c_kernel[ky + k_radius][kx + k_radius];
        }
    }
    output[out_row * width + out_col] = sum;
}

// CPU参考实现
void cpu_convolution(const float* input, float* output,
                    int width, int height,
                    const float* kernel, int kernel_size)
{
    int k_radius = kernel_size / 2;
    
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            float sum = 0.0f;
            for (int ky = -k_radius; ky <= k_radius; ++ky) {
                for (int kx = -k_radius; kx <= k_radius; ++kx) {
                    int y = row + ky;
                    int x = col + kx;
                    if (y >= 0 && y < height && x >= 0 && x < width) {
                        sum += input[y * width + x] * 
                            kernel[(ky + k_radius) * kernel_size + (kx + k_radius)];
                    }
                }
            }
            output[row * width + col] = sum;
        }
    }
}

// 生成高斯卷积核
void generate_gaussian_kernel(float* kernel, int size, float sigma)
{
    float sum = 0.0f;
    int center = size / 2;
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int x = i - center;
            int y = j - center;
            float val = exp(-(x*x + y*y)/(2*sigma*sigma));
            kernel[i*size + j] = val;
            sum += val;
        }
    }
    
    // 归一化
    for (int i = 0; i < size*size; ++i) {
        kernel[i] /= sum;
    }
}

// 验证结果
float verify(const float* ref, const float* test, int size)
{
    float max_error = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_error = fmax(max_error, fabs(ref[i] - test[i]));
    }
    return max_error;
}

int main()
{
    // 配置参数
    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    const int KERNEL_SIZE = 5;
    const float SIGMA = 1.0f;
    
    // 分配主机内存
    float *h_input, *h_output_cpu, *h_output_global, *h_output_shared, *h_kernel;
    h_input = new float[WIDTH*HEIGHT];
    h_output_cpu = new float[WIDTH*HEIGHT];
    h_output_global = new float[WIDTH*HEIGHT];
    h_output_shared = new float[WIDTH*HEIGHT];
    h_kernel = new float[KERNEL_SIZE*KERNEL_SIZE];
    
    // 初始化数据
    std::generate(h_input, h_input + WIDTH*HEIGHT, [](){ return rand() / (float)RAND_MAX; });
    generate_gaussian_kernel(h_kernel, KERNEL_SIZE, SIGMA);
    
    // 复制卷积核到常量内存
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, h_kernel, 
                KERNEL_SIZE*KERNEL_SIZE*sizeof(float)));
    
    // 分配设备内存
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, WIDTH*HEIGHT*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, WIDTH*HEIGHT*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, WIDTH*HEIGHT*sizeof(float), 
                cudaMemcpyHostToDevice));
    
    // 计算线程配置
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((WIDTH + TILE_SIZE - 1) / TILE_SIZE,
              (HEIGHT + TILE_SIZE - 1) / TILE_SIZE);
    
    // CPU计算
    auto start = std::chrono::high_resolution_clock::now();
    cpu_convolution(h_input, h_output_cpu, WIDTH, HEIGHT, h_kernel, KERNEL_SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // GPU全局内存版本
    start = std::chrono::high_resolution_clock::now();
    convolution_global<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT, KERNEL_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output_global, d_output, WIDTH*HEIGHT*sizeof(float),
                cudaMemcpyDeviceToHost));
    end = std::chrono::high_resolution_clock::now();
    float gpu_global_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // GPU共享内存版本
    size_t shared_size = (TILE_SIZE + 2*(KERNEL_SIZE/2)) * 
                        (TILE_SIZE + 2*(KERNEL_SIZE/2)) * sizeof(float);
    start = std::chrono::high_resolution_clock::now();
    convolution_shared<<<grid, block, shared_size>>>(d_input, d_output, WIDTH, HEIGHT, KERNEL_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output_shared, d_output, WIDTH*HEIGHT*sizeof(float),
                cudaMemcpyDeviceToHost));
    end = std::chrono::high_resolution_clock::now();
    float gpu_shared_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // 验证结果
    float error_global = verify(h_output_cpu, h_output_global, WIDTH*HEIGHT);
    float error_shared = verify(h_output_cpu, h_output_shared, WIDTH*HEIGHT);
    
    // 输出结果
    std::cout << "=== 性能测试结果 ===\n"
              << "CPU时间: " << cpu_time << " ms\n"
              << "GPU全局内存版本时间: " << gpu_global_time << " ms (加速比: " 
              << cpu_time/gpu_global_time << "x)\n"
              << "GPU共享内存版本时间: " << gpu_shared_time << " ms (加速比: " 
              << cpu_time/gpu_shared_time << "x)\n"
              << "最大误差(全局内存): " << error_global << "\n"
              << "最大误差(共享内存): " << error_shared << std::endl;
    
    // 释放资源
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_global;
    delete[] h_output_shared;
    delete[] h_kernel;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}
