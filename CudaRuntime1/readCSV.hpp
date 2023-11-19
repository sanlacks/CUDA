// csv_reader.h
#pragma once

#include <vector>
#include <fstream>
#include <string>

void readCSV(std::vector<double>& x, std::vector<double>& y, std::ifstream& file, int& lineCount);
void writeCSV(const std::vector<double>& data, const std::string& filename);