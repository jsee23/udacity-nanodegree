#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd::Zero(5);

  // initial covariance matrix
  P_ = MatrixXd::Zero(5, 5);
  P_ << 9., 0, 0, 0, 0,
        0, 9., 0, 0, 0,
        0, 0, .9, 0, 0,
        0, 0, 0, 3., 0,
        0, 0, 0, 0, 3.;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.6;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  is_initialized_ = false;
  time_us_ = 0.0;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_x_;
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  weights_ = VectorXd(2 * n_aug_ + 1);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) ||
      (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_)) {
    std::cout << "=> => skipped measurement" << std::endl;
    return;
  }

  if (!is_initialized_) {

    time_us_ = meas_package.timestamp_;

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      std::cout << "initilized with LASER data!" << std::endl;
      x_(0) = meas_package.raw_measurements_[0];
      x_(1) = meas_package.raw_measurements_[1];
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      std::cout << "initilized with RADAR data!" << std::endl;
      // convert coordinates
      float rho     = meas_package.raw_measurements_(0);
      float phi     = meas_package.raw_measurements_(1);
      //float rho_dot = meas_package.raw_measurements_(2);

      x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
    }

    double wt_n = lambda_ + n_aug_;
    weights_(0) = lambda_ / wt_n;
    for (int i=1; i < 2 * n_aug_ + 1; i++) {
      weights_(i) = 0.5 / wt_n;
    }

    is_initialized_ = true;
    return;
  }

  // Prediction
  float dt_in_sec = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt_in_sec);

  // Update
  if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    UpdateLidar(meas_package);
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    UpdateRadar(meas_package);
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /*** generate sigma points ***/
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  MatrixXd L = P_aug.llt().matrixL();
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2* n_aug_ + 1);
  Xsig_aug.col(0) = x_aug;
  for (int i=0; i < n_aug_; i++) {
    const MatrixXd cal = sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1)          = x_aug + cal;
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - cal;
  }

  /*** predict sigma points ***/
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
    double p_x      = Xsig_aug(0, i);
    double p_y      = Xsig_aug(1, i);
    double v        = Xsig_aug(2, i);
    double yaw      = Xsig_aug(3, i);
    double yawd     = Xsig_aug(4, i);
    double nu_a     = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    double px_p = 0.0, py_p = 0.0;

    // check again if values are too small
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    px_p += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p += 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p  += nu_a * delta_t;

    yaw_p  += 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p += nu_yawdd*delta_t;

    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  /*** convert sigma points ***/
  x_.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  P_.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = Tools::NormalizeAngle(x_diff(3));

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  std::cout << "=> lidar" << std::endl;
  VectorXd z = meas_package.raw_measurements_;
  int n_z = 2;

  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  // goal: sigma points in measurement space
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);

    z_pred += weights_(i) * Zsig.col(i);
  }

  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;
  S += R;

  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // lidar update
  Tc.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();
  VectorXd z_diff = z - z_pred;
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  double NIS = z_diff.transpose() * S.llt().matrixL() * z_diff;
  std::cout << "NIS = " << NIS << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  std::cout << "=> radar" << std::endl;
  VectorXd z = VectorXd(3);
  z << meas_package.raw_measurements_(0),
       meas_package.raw_measurements_(1),
       meas_package.raw_measurements_(2);

  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  // goal: sigma points in measurement space
  for (int i=0; i < 2 * n_aug_ + 1; i++) {

    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v   = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // rho, phi & rho_dot
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
    Zsig(1, i) = atan2(p_y, p_x);
    Zsig(2, i) = (p_x * v1 + p_y * v2) / Zsig(0, i);

    z_pred += weights_(i) * Zsig.col(i);
  }

  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = Tools::NormalizeAngle(z_diff(1));

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_* std_radr_, 0, 0,
       0, std_radphi_* std_radphi_, 0,
       0, 0, std_radrd_ * std_radrd_;
  S += R;

  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = Tools::NormalizeAngle(z_diff(1));

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = Tools::NormalizeAngle(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();
  VectorXd z_diff = z - z_pred;
  z_diff(1) = Tools::NormalizeAngle(z_diff(1));

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  //double NIS = z_diff.transpose() * S.inverse() * z_diff;
  double NIS = z_diff.transpose() * S.llt().matrixL() * z_diff;
  std::cout << "NIS = " << NIS << std::endl;
}
