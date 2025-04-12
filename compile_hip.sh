#!/bin/bash

echo "=== 编译并运行HIP高级卷积计算程序 ==="

# 设置环境变量（如果需要）
export PATH=$PATH:/opt/rocm/bin

# 编译HIP程序
echo "正在编译 convolution_hip_advanced.cu..."
hipcc -o convolution_hip_advanced convolution_hip_advanced.cu

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "编译失败，请检查错误信息。"
    exit 1
fi

echo "编译成功！正在运行程序..."
echo ""

# 运行程序
./convolution_hip_advanced

echo ""
echo "程序执行完毕。" 