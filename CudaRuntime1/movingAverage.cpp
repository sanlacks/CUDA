#include <vector>
#include "movingAverage.hpp"

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