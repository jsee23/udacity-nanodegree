#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse = VectorXd::Zero(4);

  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    std::cerr << "Invalid estimation or ground_truth data" << std::endl;
    return rmse;
  }

  for (unsigned int i=0; i < estimations.size(); i++) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  rmse /= estimations.size();
  rmse = rmse.array().sqrt();
  return rmse; 
}

float Tools::NormalizeAngle(float angle){
  return atan2(sin(angle), cos(angle) );
}
