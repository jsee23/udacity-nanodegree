#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices

  //measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  MatrixXd P_ = MatrixXd(4,4);
  P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;

  MatrixXd F_ = MatrixXd(4,4);
  F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

  MatrixXd Q_ = MatrixXd(4,4);
  VectorXd x_ = VectorXd(4);

  ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      float rho_dot = measurement_pack.raw_measurements_(2);

      ekf_.x_(0) = rho * cos(phi);
      ekf_.x_(1) = rho * sin(phi);
      ekf_.x_(2) = 0;
      ekf_.x_(3) = 0;
    }
    else  {
      /**
      Initialize state.
      */
      ekf_.x_(0) = measurement_pack.raw_measurements_[0];
      ekf_.x_(1) = measurement_pack.raw_measurements_[1];
      ekf_.x_(2) = 0;
      ekf_.x_(3) = 0;
    }

    // done initializing, no need to predict or update
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  float noise_ax = 9.0, noise_ay = 9.0;

  float delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  float delta_t_2 = delta_t * delta_t;
  float delta_t_3 = delta_t_2 * delta_t;
  float delta_t_4 = delta_t_3 * delta_t;

  ekf_.F_ << 1, 0, delta_t, 0,
             0, 1, 0, delta_t,
             0, 0, 1, 0,
             0, 0, 0, 1;

  ekf_.Q_ << delta_t_4 / 4 * noise_ax, 0,                        delta_t_3 / 2 * noise_ax, 0,
             0,                        delta_t_4 / 4 * noise_ay, 0,                        delta_t_3 / 2 * noise_ay,
             delta_t_3 / 2 * noise_ax, 0,                        delta_t_2 * noise_ax,     0,
             0,                        delta_t_3 / 2 * noise_ay, 0,                        delta_t_2 * noise_ax;


  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    std::cout << "update type => RADAR" << std::endl;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    std::cout << "update type => LASER" << std::endl;
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
