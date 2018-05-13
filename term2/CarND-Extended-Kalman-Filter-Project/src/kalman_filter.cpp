#include "kalman_filter.h"

#include <iostream>

#define EPSILON 0.00001

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd F_transposed = F_.transpose();
  P_ = F_ * P_ * F_transposed + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  const VectorXd z_predition = H_ * x_;
  const VectorXd y = z - z_predition;
  const MatrixXd H_transposed = H_.transpose();
  const MatrixXd S = H_ * P_ * H_transposed + R_;
  const MatrixXd S_inversed = S.inverse();
  const MatrixXd PH_transposed = P_ * H_transposed;
  const MatrixXd K = PH_transposed * S_inversed;

  x_ = x_ + (K * y);
  const MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  double px = x_[0], py = x_[1], vx = x_[2], vy = x_[3];
  // check if px is too small
  if (fabs(px) < EPSILON)
    px = EPSILON;

  float rho = sqrt(px * px + py * py);
  // check if rho is too small
  if (fabs(rho) < EPSILON)
    rho = EPSILON;

  float phi = atan2(py, px);
  float rho_dot = (px * vx + py * vy) / rho;

  VectorXd z_prediction = VectorXd(3);
  z_prediction << rho, phi, rho_dot;

  VectorXd y = z - z_prediction;
  while (y(1) < -M_PI || M_PI < y(1)) {
    if (y(1)  < -M_PI) {
      y(1) += 2 * M_PI;
    }
    if (y(1) > M_PI) {
      y(1) -= 2 * M_PI;
    }
  }

  const MatrixXd H_transposed = H_.transpose();
  const MatrixXd S = H_ * P_ * H_transposed + R_;
  const MatrixXd S_inversed = S.inverse();
  const MatrixXd PH = P_ * H_transposed;
  const MatrixXd K = PH * S_inversed;

  x_ = x_ + (K * y);
  const MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}
