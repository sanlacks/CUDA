// csv_reader.h
#pragma once

#include <vector>
#include <fstream>
#include <string>

void readCSV(std::vector<float>& hostData, std::ifstream& file, int& lineCount);