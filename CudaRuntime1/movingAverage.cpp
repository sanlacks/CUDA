#include <vector>
#include "movingAverage.hpp"

std::vector<double> movingAverage(const std::vector<double>& data, int windowSize) {
    int dataLength = data.size();
    std::vector<double> smoothedData(dataLength);

    for (int i = 0; i < dataLength; ++i) {
        double sum = 0.0;
        int count = 0;

        // 调整循环范围确保窗口两侧有足够的数据点
        int start = std::max(0, i - windowSize / 2);
        int end = std::min(dataLength - 1, i + windowSize / 2);

        for (int j = start; j <= end; ++j) {
            sum += data[j];
            count++;
        }

        double average = sum / count;
        smoothedData[i] = average;
    }

    return smoothedData;
}
