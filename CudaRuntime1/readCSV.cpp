#include "readCSV.hpp"
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdio.h>

using namespace std;

void readCSV(std::vector<double>& x, std::vector<double>& y, std::ifstream& file, int& lineCount) {
    std::string line;
    lineCount = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;

        // 使用逗号分隔符分割每一行的数据
        std::getline(iss, token, ',');
        double x_value = std::stod(token); // 将字符串转换为double
        x.push_back(x_value);

        std::getline(iss, token, ',');
        double y_value = std::stod(token);
        y.push_back(y_value);
    }

}

void writeCSV(const std::vector<double>& data, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (const auto& value : data) {
        file << value << "\n";  // 写入一列数据，每个数据占一行
    }

    file.close();
}