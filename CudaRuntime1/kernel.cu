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
    //for (i = 0; i < LENGTH / 2; i++)
    //{
    //    printf("i=%d\tf= %6.1fHz\tRealAmp=%3.1f\t", i, fs * i / LENGTH, CompData[i].x * 2.0 / LENGTH);
    //    printf("ImagAmp=+%3.1fi", CompData[i].y * 2.0 / LENGTH);
    //    printf("\n");
    //}

    //创建频率轴
    int n = LENGTH / 2;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < LENGTH / 2; ++i) {
        x.at(i) = fs * i / LENGTH;
        y.at(i) = abs(CompData[i].x) * 2.0 / LENGTH;

    }

    //在C++中使用matplotlibcpp绘制FFT结果的振幅谱
    plt::plot(x, y);
    //plt::pause(2);
    plt::xlim(-10, 10000);
    plt::xlabel("Frequency/hz");
    plt::ylabel("Amplitude");
    plt::grid('b');

    plt::show();
    cufftDestroy(plan);
    free(CompData);
    cudaFree(d_fftData);

    return 0;
}