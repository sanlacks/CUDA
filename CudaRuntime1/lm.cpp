#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>

using namespace Eigen;
using namespace std;

// y = k1 * x^3 + k2 * x^2 + k3 * x + k4
double func(double x, const Vector4d& para) {
    return para[0] * pow(x, 3) + para[1] * pow(x, 2) + para[2] * x + para[3];
}

// jacobi matrix
MatrixX4d jacobi(const vector<double>& x, const Vector4d& para) {
    int n = x.size();
    MatrixX4d J(n, 4);
    for (int i = 0; i < n; i++) {
        J(i, 0) = pow(x[i], 3);
        J(i, 1) = pow(x[i], 2);
        J(i, 2) = x[i];
        J(i, 3) = 1;
    }
    return J;
}

// curve fitting
Vector4d fit_curve(const vector<double>& x, const vector<double>& y, int iter_num = 10000, double eps = 1e-8) {
    int num_paras = 4;
    Vector4d para_past = Vector4d::Ones(); // parameter initialization
    VectorXd y_gj = VectorXd::Zero(x.size());
    for (int i = 0; i < x.size(); i++) {
        y_gj[i] = func(x[i], para_past);
    }
    MatrixX4d J = jacobi(x, para_past); // jacobi matrix
    VectorXd r_past = VectorXd::Zero(x.size()); // residual vector
    for (int i = 0; i < x.size(); i++) {
        r_past[i] = y[i] - y_gj[i];
    }
    cout << J.rows() << " " << J.cols() << " " << r_past.size() << endl;
    VectorXd g = J.transpose() * r_past;

    double tao = 1e-3; // (1e-8, 1)
    double u = tao * (J.transpose() * J).diagonal().maxCoeff(); // damping factor initialization
    double v = 2;

    double norm_inf = g.lpNorm<Infinity>();
    bool stop = norm_inf < eps;

    int num = 0;
    double rou = 0.0; // declare rou variable

    while (!stop && num < iter_num) {
        num++;
        while (true) {
            Matrix4d H_lm = J.transpose() * J + u * Matrix4d::Identity();
            Vector4d delt = H_lm.inverse() * g;
            double norm_2 = delt.norm();
            if (norm_2 < eps) {
                stop = true;
            }
            else {
                Vector4d para_cur = para_past + delt; // update parameter
                VectorXd y_gj_cur = VectorXd::Zero(x.size());
                for (int i = 0; i < x.size(); i++) {
                    y_gj_cur[i] = func(x[i], para_cur);
                }
                MatrixX4d J_cur = jacobi(x, para_cur);
                VectorXd r_cur = VectorXd::Zero(x.size());
                for (int i = 0; i < x.size(); i++) {
                    r_cur[i] = y[i] - y_gj_cur[i];
                }
                double rou = ((r_past.squaredNorm() - r_cur.squaredNorm()) / (delt.dot(u * delt + g)));
                if (rou > 0) {
                    para_past = para_cur;
                    r_past = r_cur;
                    J = jacobi(x, para_past);
                    g = J.transpose() * r_past;
                    stop = (g.lpNorm<Infinity>() <= eps) || (r_past.squaredNorm() <= eps);
                    u = u * max(1.0 / 3, 1 - pow(2 * rou - 1, 3));
                    v = 2;
                }
                else {
                    u *= v;
                    v *= 2;
                }
            }
            if (rou > 0 || stop) {
                break;
            }
        }
    }
    return para_past;
}