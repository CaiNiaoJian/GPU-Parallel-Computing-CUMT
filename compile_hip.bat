@echo off
echo === 编译并运行HIP高级卷积计算程序 ===

rem 设置环境变量
set PATH=%PATH%;C:\Program Files\ROCM\bin

rem 编译HIP程序
echo 正在编译 convolution_hip_advanced.cu...
hipcc -o convolution_hip_advanced convolution_hip_advanced.cu

rem 检查编译是否成功
if %ERRORLEVEL% NEQ 0 (
    echo 编译失败，请检查错误信息。
    exit /b %ERRORLEVEL%
)

echo 编译成功！正在运行程序...
echo.

rem 运行程序
convolution_hip_advanced

echo.
echo 程序执行完毕。 