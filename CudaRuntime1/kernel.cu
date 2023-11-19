#include <cufft.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <cuda_runtime.h>
#include "matplotlibcpp.h" 

#include "movingAverage.hpp"
#include "readCSV.hpp"
#include "lm.hpp"


using namespace Eigen;
using namespace std;
namespace plt = matplotlibcpp;



int main()
{
    // 打开CSV文件
    ifstream file("D:\\Codes\\VisualStudio\\data.csv");

    // 检查文件是否成功打开
    if (!file.is_open()) {
        cerr << "无法打开文件" << endl;
        return 1;
    }

    int LENGTH = 0;
    
    vector<double> x_data(LENGTH);
    vector<double> y_data(LENGTH);
    vector<double> lmy_data(LENGTH);

    //读取csv文件
    readCSV(x_data, y_data, file, LENGTH);

    // 对原始数据应用滑动平均滤波
    int windowSize1 = 5; // 设置滑动窗口大小

    int dataLength = y_data.size();

    vector<double> smoothedy_data = movingAverage(y_data, windowSize1);

    //writeCSV(smoothedy_data, "sy_data.csv");

    // lm参数计算
    Vector4d para = fit_curve(x_data, smoothedy_data);
    cout << "The optimal parameters are: " << para.transpose() << endl;

    //lm拟合
    int i = 0;
    int Nl = x_data.size();

    for (i=0;i<Nl;i++){
        
        lmy_data.push_back(func(x_data[i], para));

    }

    //writeCSV(lmy_data, "fitsy_data.csv");
    

    //plt::named_plot("Origin Data", x_data, y_data,"y");// 原始数据绘图
    
    //plt::named_plot("Smooth Data", x_data, smoothedy_data, "b");// 滤波数据绘图
    
    //plt::named_plot("LM Data", x_data, lmy_data, "r");// LM数据绘图
   
    //plt::title("Data Process");
    //plt::legend();
    //plt::show();
   
    

    

    // 将 lmy_data 复制到 GPU 内存
    double* d_lmy_data;
    cudaMalloc((void**)&d_lmy_data, Nl * sizeof(double));
    cudaMemcpy(d_lmy_data, lmy_data.data(), Nl * sizeof(double), cudaMemcpyHostToDevice);
    
    // 创建 CUFFT 计划
    cufftHandle plan;
    cufftPlan1d(&plan, Nl, CUFFT_Z2Z, 1);

    // 执行 FFT
    cufftExecZ2Z(plan, (cufftDoubleComplex*)d_lmy_data, (cufftDoubleComplex*)d_lmy_data, CUFFT_FORWARD);

    // 将结果传回主机
    cudaMemcpy(lmy_data.data(), d_lmy_data, Nl * sizeof(double), cudaMemcpyDeviceToHost);

    // 销毁 CUFFT 计划和 GPU 内存
    cufftDestroy(plan);
    cudaFree(d_lmy_data);

    //double fs = 1 / 0.0001220703125; //采样率
    double fs = 1000; //采样率

    
    // 创建频率轴
    int n = Nl/2;
    std::vector<double> x(n), y(n);
    
    for (int i = 0; i < Nl/2; ++i){

        x.at(i) = fs * i / Nl;
        y.at(i) = abs(lmy_data[i]) * 2.0 / Nl;

    }
    // 使用max_element 查找最大值的迭代器
    auto maxElement = max_element(y.begin(), y.end());
   
    cout << "最大值是: " << *maxElement << std::endl;
   
    

    // 滑动平均滤波
    int windowSize2 = 5; // 设置滑动窗口大小
    std::vector<double> smoothedY = movingAverage(y, windowSize2);

    // 使用matplotlibcpp绘制结果   
    plt::plot(x, smoothedY); // 使用滤波后的数据进行绘图
    //plt::xlim(-100, 3000);
    //plt::xlabel("Frequency/hz");
    //plt::ylabel("Amplitude");
   // plt::grid('b');
    plt::show();
    

    return 0;
}