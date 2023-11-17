#ifndef LM_H
#define LM_H

#include <Eigen/Dense>
#include <vector>



// y = k1 * x^3 + k2 * x^2 + k3 * x + k4
double func(double x, const Eigen::Vector4d& para);

// Jacobi matrix
Eigen::MatrixX4d jacobi(const std::vector<double>& x, const Eigen::Vector4d& para);

// Curve fitting
Eigen::Vector4d fit_curve(const std::vector<double>& x, const std::vector<double>& y, int iter_num = 10000, double eps = 1e-8);

#endif 
