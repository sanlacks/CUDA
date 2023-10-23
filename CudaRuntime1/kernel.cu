#include <math.h>
#include <vector>
#include <cufft.h>
#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex>
#include <iostream>
#include <cuda_runtime.h>
#include "matplotlibcpp.h" 
#include "device_launch_parameters.h"

using namespace std;
namespace plt = matplotlibcpp;


//读取CSV文件并返回行数
void readCSV(std::vector<float>& hostData, std::ifstream& file, int& lineCount) {
    std::string line;
    lineCount = 0;

    while (std::getline(file, line)) {
        lineCount++;
        hostData.push_back(std::stof(line));
    }
}

// 滑动平均滤波函数
std::vector<float> movingAverage(const std::vector<float>& data, int windowSize) {
    std::vector<float> smoothedData;
    int dataLength = data.size();

    for (int i = 0; i < dataLength; ++i) {
        float sum = 0.0f;
        int count = 0;

        for (int j = i - windowSize / 2; j <= i + windowSize / 2; ++j) {
            if (j >= 0 && j < dataLength) {
                sum += data[j];
                count++;
            }
        }

        float average = sum / count;
        smoothedData.push_back(average);
    }

    return smoothedData;
}

int main()
{
    // 打开CSV文件
    std::ifstream file("D:\\Codes\\VisualStudio\\signal13.csv");

    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }

    int LENGTH = 0;
    vector<float> Data(LENGTH);

    //读取csv文件
    readCSV(Data, file, LENGTH);
    // 对原始数据应用滑动平均滤波
    int windowSize1 = 10; // 设置滑动窗口大小
    std::vector<float> smoothedData = movingAverage(Data, windowSize1);
    

    // 分配和传输数据到CUDA设备
    cufftComplex* CompData = (cufftComplex*)malloc(LENGTH * sizeof(cufftComplex));//allocate memory for the data in host
 
    for (int i = 0; i < LENGTH; i++)
    {
        CompData[i].x = Data[i];
        CompData[i].y = 0;
    }

    cufftComplex* d_fftData;
    cudaMalloc((void**)&d_fftData, LENGTH * sizeof(cufftComplex));// allocate memory for the data in device
    cudaMemcpy(d_fftData, CompData, LENGTH * sizeof(cufftComplex), cudaMemcpyHostToDevice);// copy data from host to device

    // 创建cuFFT计划
    cufftHandle plan;// cuda library function handle
    cufftPlan1d(&plan, LENGTH, CUFFT_C2C, 1);//declaration

    // 执行FFT
    cufftExecC2C(plan, (cufftComplex*)d_fftData, (cufftComplex*)d_fftData, CUFFT_FORWARD);//execute
    cudaDeviceSynchronize();//wait to be done

    // 传输FFT结果到主机
    cudaMemcpy(CompData, d_fftData, LENGTH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);// copy the result from device to host


    //double fs = 1 / 0.0001220703125; //采样率
    double fs = 100000; //采样率

    // 创建频率轴
    int n = LENGTH / 2;
    std::vector<float> x(n), y(n);
    for (int i = 0; i < LENGTH / 2; ++i)
    {
        x.at(i) = fs * i / LENGTH;
        y.at(i) = abs(CompData[i].x) * 2.0 / LENGTH;
    }

    // 滑动平均滤波
    int windowSize2 = 5; // 设置滑动窗口大小
    std::vector<float> smoothedY = movingAverage(y, windowSize2);

    // 使用matplotlibcpp绘制结果
    plt::plot(x, smoothedY); // 使用滤波后的数据进行绘图
    plt::xlim(-100, 3000);
    plt::xlabel("Frequency/hz");
    plt::ylabel("Amplitude");
    plt::grid('b');

    plt::show();
    cufftDestroy(plan);
    free(CompData);
    cudaFree(d_fftData);

    return 0;
}