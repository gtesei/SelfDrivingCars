#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */

// Lidar has 2 degree of freedom --> hence, Pr(NIS > 5.99) ~ 5%
double NIS_Radar_tot = 0;
double NIS_Radar_top_5_perc = 0;

// Radar has 3 degree of freedom --> hence, Pr(NIS > 7.81) ~ 5%
double NIS_Lidar_tot = 0;
double NIS_Lidar_top_5_perc = 0;

UKF::UKF() {
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  ///* initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  ///* if this is false, laser measurements will be ignored (except for init)
  use_laser_ = true;

  ///* if this is false, radar measurements will be ignored (except for init)
  use_radar_ = true;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  x_ = VectorXd(5);
  x_.fill(0.0);

  ///* state covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0,0,
        0, 1, 0, 0,0,
        0, 0, 1, 0,0,
        0, 0, 0, 1,0,
        0, 0, 0, 0,1;

  ///* State dimension
  n_x_ = x_.rows();

  ///* Augmented state dimension
  n_aug_ = n_x_ + 2;

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  ///* predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.57;

  ///* Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  ///* Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  ///* Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  ///* Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  ///* Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  ///* weights_ of sigma points
  weights_ = VectorXd(2*n_aug_+1);
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights_
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
      /**
      TODO:
        * Initialize the state ekf_.x_ with the first measurement.
        * Create the covariance matrix.
        * Remember: you'll need to convert radar from polar to cartesian coordinates.
      */
      // first measurement
      cout << "UKF: " << endl;

      if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        /**
        Convert radar from polar to cartesian coordinates and initialize state.
        */
          float ro     = measurement_pack.raw_measurements_[0];

          float phi    = measurement_pack.raw_measurements_[1];
          //angle normalization
          while (phi > M_PI)  phi -= 2. * M_PI;
          while (phi < -M_PI) phi += 2. * M_PI;

          float ro_dot = measurement_pack.raw_measurements_[2];

          x_(0) = ro     * cos(phi);
          x_(1) = ro     * sin(phi);
          x_(2) = 0;
          x_(3) = phi;
          x_(4) = 0;
      }
      else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
        /**
        Initialize state.
        */
          x_(0) = measurement_pack.raw_measurements_[0];
          x_(1) = measurement_pack.raw_measurements_[1];

          float phi = atan2(x_(1), x_(0));
          //angle normalization
          while (phi > M_PI)  phi -= 2. * M_PI;
          while (phi < -M_PI) phi += 2. * M_PI;

          x_(2) = 0;
          x_(3) = phi;
          x_(4) = 0;
      }

      // done initializing, no need to predict or update
      time_us_ = measurement_pack.timestamp_;
      is_initialized_ = true;
      cout << "is_initialized. " << endl;
      return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    float dt = (measurement_pack.timestamp_ - time_us_) / 1000000.0;    //dt - expressed in seconds
    time_us_ = measurement_pack.timestamp_;

    Prediction(dt);



    /*****************************************************************************
     *  Update
     ****************************************************************************/

     if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
         // Radar updates
         if (use_radar_) {
            UpdateRadar(measurement_pack);
         }
     } else {
         // Laser updates
         if (use_laser_) {
            UpdateLidar(measurement_pack);
          }
     }

     // print the output
     // cout << "x_ = " << x_ << endl;
     // cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  /*******************************************************************************
   * 1. Generate Sigma Points
   ******************************************************************************/

   //create augmented mean vector
   VectorXd x_aug = VectorXd(n_aug_);

   //create augmented state covariance
   MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

   //create sigma point matrix
   MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

   //create augmented mean state
   x_aug.head(5) = x_;
   x_aug(5) = 0;
   x_aug(6) = 0;

   //create augmented covariance matrix
   P_aug.fill(0.0);
   P_aug.topLeftCorner(5,5) = P_;
   P_aug(5,5) = std_a_*std_a_;
   P_aug(6,6) = std_yawdd_*std_yawdd_;

   //create square root matrix
   MatrixXd L = P_aug.llt().matrixL();

   //create augmented sigma points
   Xsig_aug.col(0)  = x_aug;
   for (int i = 0; i< n_aug_; i++) {
      Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
      Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
   }

  /*******************************************************************************
   * 2. Predict Sigma Points
   ******************************************************************************/

   for (int i = 0; i< 2*n_aug_+1; i++) {
       //extract values for better readability
       double p_x = Xsig_aug(0,i);
       double p_y = Xsig_aug(1,i);
       double v = Xsig_aug(2,i);
       double yaw = Xsig_aug(3,i);
       double yawd = Xsig_aug(4,i);
       double nu_a = Xsig_aug(5,i);
       double nu_yawdd = Xsig_aug(6,i);

       //predicted state values
       double px_p, py_p;

       //avoid division by zero
       if (fabs(yawd) > 0.001) {
           px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
           py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
       } else {
           px_p = p_x + v*delta_t*cos(yaw);
           py_p = p_y + v*delta_t*sin(yaw);
       }

       double v_p = v; // constant velocity
       double yaw_p = yaw + yawd*delta_t;
       double yawd_p = yawd; // constant turning rate

       //add noise
       px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
       py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
       v_p = v_p + nu_a*delta_t;

       yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
       yawd_p = yawd_p + nu_yawdd*delta_t;

       //write predicted sigma point into right column
       Xsig_pred_(0,i) = px_p;
       Xsig_pred_(1,i) = py_p;
       Xsig_pred_(2,i) = v_p;
       Xsig_pred_(3,i) = yaw_p;
       Xsig_pred_(4,i) = yawd_p;
     }

  /*******************************************************************************
   * 3. Predict Mean and Covariance
   ******************************************************************************/

     //predicted state mean
     x_.fill(0.0);
     for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
       x_ = x_ + weights_(i) * Xsig_pred_.col(i);
     }

     //predicted state covariance matrix
     P_.fill(0.0);
     for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
       // state difference
       VectorXd x_diff = Xsig_pred_.col(i) - x_;
       //angle normalization
       while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
       while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

       P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
     }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage measurement_pack) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  /*******************************************************************************
   * 1. Predict Measurement
   ******************************************************************************/

    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 2;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);


        // measurement model
        Zsig(0,i) = p_x;                        //p_x
        Zsig(1,i) = p_y;                        //p_y
      }

      //mean predicted measurement
      z_pred.fill(0.0);
      for (int i=0; i < 2*n_aug_+1; i++) {
          z_pred = z_pred + weights_(i) * Zsig.col(i);
      }

      //measurement covariance matrix S
      S.fill(0.0);
      for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        S = S + weights_(i) * z_diff * z_diff.transpose();
      }

      //add measurement noise covariance matrix
      MatrixXd R = MatrixXd(n_z,n_z);
      R << std_laspx_*std_laspx_, 0,
           0, std_laspy_*std_laspy_;

      S = S + R;

    /*******************************************************************************
     * 2. Update State
     ******************************************************************************/

     //create example vector for incoming radar measurement
     VectorXd z = VectorXd(n_z);
     z << measurement_pack.raw_measurements_[0],
          measurement_pack.raw_measurements_[1];

     //create matrix for cross correlation Tc
     MatrixXd Tc = MatrixXd(n_x_, n_z);

     //calculate cross correlation matrix
     Tc.fill(0.0);
     for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
         //residual
         VectorXd z_diff = Zsig.col(i) - z_pred;

         // state difference
         VectorXd x_diff = Xsig_pred_.col(i) - x_;

         Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
       }

       //Kalman gain K;
       MatrixXd K = Tc * S.inverse();

       //residual
       VectorXd z_diff = z - z_pred;

       //update state mean and covariance matrix
       x_ = x_ + K * z_diff;
       P_ = P_ - K*S*K.transpose();
    
       //// -------> calculate NIS
       // Lidar has 2 degree of freedom --> hence, Pr(NIS > 5.99) ~ 5%
       double NIS = z_diff.transpose() * S.inverse() * z_diff;
       if (NIS > 5.99) {
           NIS_Lidar_top_5_perc += 1;
       }
       NIS_Lidar_tot += 1;
       double observed_top_5_perc = NIS_Lidar_top_5_perc / NIS_Lidar_tot;
       cout << "NIS Lidar: " << NIS << " ---> observed_top_5_perc: " << observed_top_5_perc << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage measurement_pack) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  /*******************************************************************************
   * 1. Predict Measurement
   ******************************************************************************/

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

      // extract values for better readibility
      double p_x = Xsig_pred_(0,i);
      double p_y = Xsig_pred_(1,i);
      double v  = Xsig_pred_(2,i);
      double yaw = Xsig_pred_(3,i);

      double v1 = cos(yaw)*v;
      double v2 = sin(yaw)*v;

      // measurement model
      Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r

      float phi = atan2(p_y,p_x);                                 //phi
      //angle normalization
      while (phi> M_PI) phi-=2.*M_PI;
      while (phi<-M_PI) phi+=2.*M_PI;
      Zsig(1,i) = phi;

      Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }

    //mean predicted measurement
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
      //residual
      VectorXd z_diff = Zsig.col(i) - z_pred;

      //angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R << std_radr_*std_radr_, 0, 0,
         0, std_radphi_*std_radphi_, 0,
         0, 0,std_radrd_*std_radrd_;

    S = S + R;

  /*******************************************************************************
   * 2. Update State
   ******************************************************************************/

   //create example vector for incoming radar measurement
    VectorXd z = VectorXd(n_z);
    z << measurement_pack.raw_measurements_[0],
         measurement_pack.raw_measurements_[1],
         measurement_pack.raw_measurements_[2];

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
      }

      //Kalman gain K;
      MatrixXd K = Tc * S.inverse();

      //residual
      VectorXd z_diff = z - z_pred;

      //angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      //update state mean and covariance matrix
      x_ = x_ + K * z_diff;
      P_ = P_ - K*S*K.transpose();
    
      //// -------> calculate NIS
      // Radar has 3 degree of freedom --> hence, Pr(NIS > 7.81) ~ 5%
      double NIS = z_diff.transpose() * S.inverse() * z_diff;
      if (NIS > 7.81) {
         NIS_Radar_top_5_perc += 1;
      }
      NIS_Radar_tot += 1;
      double observed_top_5_perc = NIS_Radar_top_5_perc / NIS_Radar_tot;
      cout << "NIS Radar: " << NIS << " ---> observed_top_5_perc: " << observed_top_5_perc << endl;
}