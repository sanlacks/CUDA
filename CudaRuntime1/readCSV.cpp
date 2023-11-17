#include "readCSV.hpp"
void readCSV(std::vector<float>& hostData, std::ifstream& file, int& lineCount) {
    std::string line;
    lineCount = 0;

    while (std::getline(file, line)) {
        lineCount++;
        hostData.push_back(std::stof(line));
    }
}