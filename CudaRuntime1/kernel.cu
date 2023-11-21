#include <cufft.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <cuda_runtime.h>
#include "matplotlibcpp.h" 
#include "fftw3.h"
#include "movingAverage.hpp"
#include "readCSV.hpp"
#include "lm.hpp"


using namespace Eigen;
using namespace std;
namespace plt = matplotlibcpp;



int main()
{
    // 打开CSV文件
    ifstream file1("D:\\Codes\\VisualStudio\\datas\\data1.csv");
    ifstream file2("D:\\Codes\\VisualStudio\\datas\\data2.csv");
    ifstream file3("D:\\Codes\\VisualStudio\\datas\\data3.csv");
    ifstream file4("D:\\Codes\\VisualStudio\\datas\\data4.csv");


    // 检查文件是否成功打开
    //if (!file.is_open()) {
    //    cerr << "无法打开文件" << endl;
    //    return 1;
    //}

    int LENGTH = 0;
    
    vector<double> x_data1(LENGTH);
    vector<double> y_data1(LENGTH);
    vector<double> lmy_data1(LENGTH);
    //vector<double> lmy_data11(LENGTH);

    vector<double> x_data2(LENGTH);
    vector<double> y_data2(LENGTH);
    vector<double> lmy_data2(LENGTH);

    vector<double> x_data3(LENGTH);
    vector<double> y_data3(LENGTH);
    vector<double> lmy_data3(LENGTH);

    vector<double> x_data4(LENGTH);
    vector<double> y_data4(LENGTH);
    vector<double> lmy_data4(LENGTH);

    //读取csv文件
    readCSV(x_data1, y_data1, file1, LENGTH);

    readCSV(x_data2, y_data2, file2, LENGTH);

    readCSV(x_data3, y_data3, file3, LENGTH);

    readCSV(x_data4, y_data4, file4, LENGTH);

    //plt::named_plot("Data 1", x_data1, y_data1);// Data1数据绘图

    //plt::named_plot("Data 2", x_data2, y_data2);// Data2数据绘图

    //plt::named_plot("Data 3", x_data3, y_data3);// Data3数据绘图

    //plt::named_plot("Data 3", x_data4, y_data4);// Data4数据绘图
    //plt::title("Data Signal");
   
    //plt::show();

    // 对原始数据应用滑动平均滤波
    int windowSize1 = 5; // 设置滑动窗口大小

    int dataLength = y_data1.size();

    vector<double> smoothedy_data1 = movingAverage(y_data1, windowSize1);

    vector<double> smoothedy_data2 = movingAverage(y_data2, windowSize1);

    vector<double> smoothedy_data3 = movingAverage(y_data3, windowSize1);

    vector<double> smoothedy_data4 = movingAverage(y_data4, windowSize1);

    //writeCSV(smoothedy_data, "sy_data.csv");

    // lm参数计算
    Vector4d para1 = fit_curve(x_data1, smoothedy_data1);
    //cout << "Data 1's parameters are: " << para1.transpose() << endl;
    Vector4d para2 = fit_curve(x_data2, smoothedy_data2);
    //cout << "Data 2's parameters are: " << para2.transpose() << endl;
    Vector4d para3 = fit_curve(x_data3, smoothedy_data3);
    //cout << "Data 3's parameters are: " << para3.transpose() << endl;
    Vector4d para4 = fit_curve(x_data4, smoothedy_data4);
    //cout << "Data 4's parameters are: " << para4.transpose() << endl;

    //lm拟合
    int i = 0;
    int Nl = x_data1.size();

    for (i=0;i<Nl;i++){   
        lmy_data1.push_back(func(x_data1[i], para1));
    }

    for (i = 0; i < Nl; i++) {
        lmy_data2.push_back(func(x_data2[i], para2));
    }

    for (i = 0; i < Nl; i++) {
        lmy_data3.push_back(func(x_data3[i], para3));
    }

    for (i = 0; i < Nl; i++) {
        lmy_data4.push_back(func(x_data4[i], para4));
    }

    //writeCSV(lmy_data, "fitsy_data.csv");
    

    //plt::named_plot("Origin Data", x_data1, y_data1,":");// 原始数据绘图
    //plt::named_plot("Smooth Data", x_data1, smoothedy_data1, "--");// 滤波数据绘图
    //plt::named_plot("LM Data", x_data1, lmy_data1, "r");// LM数据绘图
    //plt::title("Data 1 Process");
    //plt::legend();
    //plt::show();
   
    //plt::named_plot("Origin Data", x_data2, y_data2, ":");// 原始数据绘图
    //plt::named_plot("Smooth Data", x_data2, smoothedy_data2, "--");// 滤波数据绘图
    //plt::named_plot("LM Data", x_data2, lmy_data2, "r");// LM数据绘图
    //plt::title("Data 2 Process");
    //plt::legend();
    //plt::show();

    //plt::named_plot("Origin Data", x_data3, y_data3, ":");// 原始数据绘图
    //plt::named_plot("Smooth Data", x_data3, smoothedy_data3, "--");// 滤波数据绘图
    //plt::named_plot("LM Data", x_data3, lmy_data3, "r");// LM数据绘图
    //plt::title("Data 3 Process");
    //plt::legend();
    //plt::show();

    //plt::named_plot("Origin Data", x_data4, y_data4, ":");// 原始数据绘图
    //plt::named_plot("Smooth Data", x_data4, smoothedy_data4, "--");// 滤波数据绘图
    //plt::named_plot("LM Data", x_data4, lmy_data4, "r");// LM数据绘图
    //plt::title("Data 4 Process");
    //plt::legend();
    //plt::show();
 

    // 将 lmy_data1 复制到 GPU 内存
    double* d_lmy_data1;
    cudaMalloc((void**)&d_lmy_data1, Nl * sizeof(double));
    cudaMemcpy(d_lmy_data1, lmy_data1.data(), Nl * sizeof(double), cudaMemcpyHostToDevice);
    // 创建 CUFFT 计划
    cufftHandle plan;
    cufftPlan1d(&plan, Nl, CUFFT_Z2Z, 1);
    // 执行 FFT
    cufftExecZ2Z(plan, (cufftDoubleComplex*)d_lmy_data1, (cufftDoubleComplex*)d_lmy_data1, CUFFT_FORWARD);
    // 将结果传回主机
    cudaMemcpy(lmy_data1.data(), d_lmy_data1, Nl * sizeof(double), cudaMemcpyDeviceToHost);

    writeCSV(lmy_data1, "lmy_data1_fft.csv");

    // 将 lmy_data2 复制到 GPU 内存
    double* d_lmy_data2;
    cudaMalloc((void**)&d_lmy_data2, Nl * sizeof(double));
    cudaMemcpy(d_lmy_data2, lmy_data2.data(), Nl * sizeof(double), cudaMemcpyHostToDevice);
    // 创建 CUFFT 计划
    
    //cufftPlan1d(&plan1, Nl, CUFFT_Z2Z, 1);
    // 执行 FFT
    cufftExecZ2Z(plan, (cufftDoubleComplex*)d_lmy_data2, (cufftDoubleComplex*)d_lmy_data2, CUFFT_FORWARD);
    // 将结果传回主机
    cudaMemcpy(lmy_data2.data(), d_lmy_data2, Nl * sizeof(double), cudaMemcpyDeviceToHost);
    
    // 将 lmy_data3 复制到 GPU 内存
    double* d_lmy_data3;
    cudaMalloc((void**)&d_lmy_data3, Nl * sizeof(double));
    cudaMemcpy(d_lmy_data3, lmy_data3.data(), Nl * sizeof(double), cudaMemcpyHostToDevice);
    // 创建 CUFFT 计划
    
    //cufftPlan1d(&plan1, Nl, CUFFT_Z2Z, 1);
    // 执行 FFT
    cufftExecZ2Z(plan, (cufftDoubleComplex*)d_lmy_data3, (cufftDoubleComplex*)d_lmy_data3, CUFFT_FORWARD);
    // 将结果传回主机
    cudaMemcpy(lmy_data3.data(), d_lmy_data3, Nl * sizeof(double), cudaMemcpyDeviceToHost);
    
    // 将 lmy_data4 复制到 GPU 内存
    double* d_lmy_data4;
    cudaMalloc((void**)&d_lmy_data4, Nl * sizeof(double));
    cudaMemcpy(d_lmy_data4, lmy_data4.data(), Nl * sizeof(double), cudaMemcpyHostToDevice);
    // 创建 CUFFT 计划
    
    //cufftPlan1d(&plan1, Nl, CUFFT_Z2Z, 1);
    // 执行 FFT
    cufftExecZ2Z(plan, (cufftDoubleComplex*)d_lmy_data4, (cufftDoubleComplex*)d_lmy_data4, CUFFT_FORWARD);
    // 将结果传回主机
    cudaMemcpy(lmy_data4.data(), d_lmy_data4, Nl * sizeof(double), cudaMemcpyDeviceToHost);

    // 销毁 CUFFT 计划和 GPU 内存
    cufftDestroy(plan);
    
    cudaFree(d_lmy_data1);
    cudaFree(d_lmy_data2);
    cudaFree(d_lmy_data3);
    cudaFree(d_lmy_data4);

    //double fs = 1 / 0.0001220703125; //采样率
    double fs = 1000; //采样率

    
    // 创建频率轴1
    int n = Nl/2;
    std::vector<double> x1(n), y1(n);
    std::vector<double> x2(n), y2(n);
    std::vector<double> x3(n), y3(n);
    std::vector<double> x4(n), y4(n);
    
    for (int i = 0; i < Nl/2; ++i){

        x1.at(i) = fs * i / Nl;
        y1.at(i) = abs(lmy_data1[i]) * 2.0 / Nl;

    }
    // 使用max_element 查找最大值的迭代器
    auto maxElement1 = max_element(y1.begin(), y1.end());
   
    //cout << "最大值是: " << *maxElement1 << std::endl;

    // 创建频率轴2
    for (int i = 0; i < Nl / 2; ++i) {

        x2.at(i) = fs * i / Nl;
        y2.at(i) = abs(lmy_data2[i]) * 2.0 / Nl;

    }
    // 使用max_element 查找最大值的迭代器
    auto maxElement2 = max_element(y2.begin(), y2.end());

    //cout << "最大值是: " << *maxElement2 << std::endl;

    // 创建频率轴3
    for (int i = 0; i < Nl / 2; ++i) {

        x3.at(i) = fs * i / Nl;
        y3.at(i) = abs(lmy_data3[i]) * 2.0 / Nl;

    }
    // 使用max_element 查找最大值的迭代器
    auto maxElement3 = max_element(y3.begin(), y3.end());

    //cout << "最大值是: " << *maxElement3 << std::endl;

    // 创建频率轴4
    for (int i = 0; i < Nl / 2; ++i) {

        x4.at(i) = fs * i / Nl;
        y4.at(i) = abs(lmy_data4[i]) * 2.0 / Nl;

    }
    // 使用max_element 查找最大值的迭代器
    auto maxElement4 = max_element(y4.begin(), y4.end());

    //cout << "最大值是: " << *maxElement4 << std::endl;
    

    // 滑动平均滤波
    int windowSize2 = 5; // 设置滑动窗口大小
   vector<double> smoothedY1 = movingAverage(y1, windowSize2);

   vector<double> smoothedY2 = movingAverage(y2, windowSize2);

   vector<double> smoothedY3 = movingAverage(y3, windowSize2);

   vector<double> smoothedY4 = movingAverage(y4, windowSize2);




    // 使用matplotlibcpp绘制结果   
    plt::plot(x1, smoothedY1); // 使用滤波后的数据进行绘图
    //plt::xlim(-100, 3000);
    plt::xlabel("Frequency/hz");
    plt::ylabel("Amplitude");
   // plt::grid('b');
    plt::title("Data 1 Frequency-Amplitude");
    plt::show();

    plt::plot(x2, smoothedY2);
    plt::xlabel("Frequency/hz");
    plt::ylabel("Amplitude");
    plt::title("Data 2 Frequency-Amplitude");
    //plt::show();

    plt::plot(x3, smoothedY3);
    plt::xlabel("Frequency/hz");
    plt::ylabel("Amplitude");
    plt::title("Data 3 Frequency-Amplitude");
    //plt::show();

    plt::plot(x4, smoothedY4);
    plt::xlabel("Frequency/hz");
    plt::ylabel("Amplitude");
    plt::title("Data 4 Frequency-Amplitude");
    //plt::show();

    Eigen::Matrix2d A(2, 2);
    Eigen::Vector2d b(2, 1);

    A(0, 0) = *maxElement1;
    A(0, 1) = *maxElement2;
    A(1, 0) = *maxElement3;
    A(1, 1) = *maxElement4;

    b(0, 0) = *maxElement3 + *maxElement1;
    b(1, 0) = *maxElement4 + *maxElement2;
    cout << "Matrix A:\n" << A << endl;
    cout << "Vector b:\n" << b << endl;

    double a, D;
    double Omega_n = 60;//hz
    double Delta_t = 0.001;//s
    double Ne;

    Eigen::Vector2d x = A.colPivHouseholderQr().solve(b);
    a = x(0, 0);
    D = x(1, 0) / (2 + a);
    Ne = pow(cos(-a / 2), -1) / (Omega_n * Delta_t);

    cout << "The a is:\n" << a << endl;
    cout << "The D is:\n" << D << endl;
    cout << "The Ne is:\n" << Ne << endl;

    return 0;
}