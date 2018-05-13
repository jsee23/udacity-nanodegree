#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

#define EPSILON 0.00001

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse = VectorXd::Zero(4);

  if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
    std::cout << __FUNCTION__ << ":\tError - invalid estimation or ground_truth data"
              << std::endl;
    return rmse;
  }

  for (unsigned int i=0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd hj = MatrixXd::Zero(3,4);

  double px = x_state(0), py = x_state(1), vx = x_state(2), vy = x_state(3);
  double d1 = px * px + py * py;
  double d2 = sqrt(d1);
  double d3 = d1 * d2;

  if (fabs(d1) < EPSILON || fabs(d2) < EPSILON || fabs(d3) < EPSILON) {
    std::cout << __FUNCTION__ << ":\tError - division by zero"
              << std::endl;
    d1 = EPSILON;
    return hj;
  }

  hj << px / d2,                       py / d2,                       0,       0,
        -py / d1,                      px / d1,                       0,       0,
        py * (vx * py - vy * px) / d3, px * (vy * px - vx * py) / d3, px / d2, py / d2;
  return hj;
}
