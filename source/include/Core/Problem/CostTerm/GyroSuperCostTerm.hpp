#pragma once

#include <memory>
#include <Eigen/Core>
#include <vector>

#include <tbb/parallel_reduce.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/parallel_for.h>
#include <tbb/mutex.h>

#include "Core/Trajectory/ConstVelocity/Helper.hpp"
#include "Core/Evaluable/se3/Se3StateVariable.hpp"
#include "Core/Evaluable/StateVariable.hpp"
#include "Core/Evaluable/vspace/VSpaceStateVar.hpp"
#include "Core/Problem/CostTerm/BaseCostTerm.hpp"
#include "Core/Problem/CostTerm/IMUSuperCostTerm.hpp"
#include "Core/Problem/LossFunc/LossFunc.hpp"
#include "Core/Problem/NoiseModel/StaticNoiseModel.hpp"
#include "Core/Problem/Problem.hpp"
#include "Core/Trajectory/ConstVelocity/Interface.hpp"
#include "Core/Trajectory/Time.hpp"

namespace slam {
    namespace problem {
        namespace costterm {

            // -----------------------------------------------------------------------------
            /**
             * @class GyroSuperCostTerm
             * @brief Implements a cost term for gyro bias estimation in trajectory optimization.
             *
             * This cost term models gyroscope bias drift over time within a **factor graph optimization framework**.  
             * It uses a **weighted least squares formulation** to estimate the error in gyroscope measurements.
             *
             * **Mathematical Formulation:**
             * The cost function follows a **weighted least squares approach**:
             * \f[
             * J = \rho \left( \left( R^{-\frac{1}{2}} (b_{gyro} - \hat{b}_{gyro}) \right)^T 
             * \left( R^{-\frac{1}{2}} (b_{gyro} - \hat{b}_{gyro}) \right) \right)
             * \f]
             * where:
             * - \( J \) is the cost function.
             * - \( \rho(\cdot) \) is a robust loss function.
             * - \( b_{gyro} \) is the estimated gyroscope bias.
             * - \( \hat{b}_{gyro} \) is the prior bias estimate.
             * - \( R \) is the noise covariance matrix.
             *
             * This term contributes to **state estimation, sensor calibration, and motion estimation**
             * in SLAM (Simultaneous Localization and Mapping) and **sensor fusion** frameworks.
             *
             * **Usage in Optimization:**
             * - Computes the **cost contribution** of gyroscope bias errors.
             * - Contributes to the **Gauss-Newton approximation** of the Hessian matrix.
             * - Supports **robust cost functions** (e.g., L2, Cauchy, Geman-McClure).
             *
             * @note This term is often combined with IMU factors for **full state estimation**.
             */
            class GyroSuperCostTerm : public BaseCostTerm {
            public:
                
                // -----------------------------------------------------------------------------
                /**
                 * @enum LOSS_FUNC
                 * @brief Defines available robust loss functions for gyro bias optimization.
                 *
                 * Loss functions are used to **robustify** optimization against outliers.
                 * Available options:
                 * - `L2`: Least squares (no robustness).
                 * - `DCS`: Dynamic Covariance Scaling (adaptive robust cost).
                 * - `CAUCHY`: Cauchy loss (reduces outlier influence).
                 * - `GM`: Geman-McClure (heavily penalizes large residuals).
                 *
                 * @see Options
                 */
                enum class LOSS_FUNC { L2, DCS, CAUCHY, GM };

                // -----------------------------------------------------------------------------
                /**
                 * @struct Options
                 * @brief Configuration settings for `GyroSuperCostTerm`.
                 *
                 * This structure allows customization of the cost term, including:
                 * - **Threading**: Controls the number of threads for computation.
                 * - **Loss function settings**: Specifies the robust loss function for bias errors.
                 * - **Gyroscope bias prior information**: Defines IMU position offset in the system.
                 * - **SE(2) mode**: Enables 2D-only optimization.
                 */
                struct Options {
                    LOSS_FUNC gyro_loss_func = LOSS_FUNC::CAUCHY;
                    double gyro_loss_sigma = 0.1;
                    Eigen::Vector3d r_imu_ang = Eigen::Vector3d::Zero();
                    bool se2 = false;
                };

                // -----------------------------------------------------------------------------
                using Ptr = std::shared_ptr<GyroSuperCostTerm>;
                using ConstPtr = std::shared_ptr<const GyroSuperCostTerm>;
                using PoseType = liemath::se3::Transformation;
                using VelType = Eigen::Matrix<double, 6, 1>;
                using BiasType = Eigen::Matrix<double, 6, 1>;
                using Interface = slam::traj::const_vel::Interface;
                using Variable = slam::traj::const_vel::Variable;
                using Time = slam::traj::Time;
                using Matrix12d = Eigen::Matrix<double, 12, 12>;
                using Matrix6d = Eigen::Matrix<double, 6, 6>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create an instance of `GyroSuperCostTerm`.
                 *
                 * This method returns a shared pointer to an initialized cost term.
                 * It is preferred over direct instantiation to facilitate shared ownership.
                 *
                 * @param interface  Pointer to the **trajectory interface** containing system states.
                 * @param time1      Start time for bias estimation.
                 * @param time2      End time for bias estimation.
                 * @param bias1      Bias state variable at time1.
                 * @param bias2      Bias state variable at time2.
                 * @param options    Struct containing additional cost term options.
                 *
                 * @return Shared pointer to the constructed cost term.
                 */
                static Ptr MakeShared(const Interface::ConstPtr &interface,
                                      const Time &time1, const Time &time2,
                                      const slam::eval::Evaluable<BiasType>::ConstPtr &bias1,
                                      const slam::eval::Evaluable<BiasType>::ConstPtr &bias2,
                                      const Options &options);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs the gyroscope cost term for optimization.
                 *
                 * This constructor initializes the cost term using the provided system states
                 * and **prior bias estimates** from IMU sensor readings.
                 *
                 * @param interface  Pointer to the **trajectory interface** containing system states.
                 * @param time1      Start time for bias estimation.
                 * @param time2      End time for bias estimation.
                 * @param bias1      Bias state variable at time1.
                 * @param bias2      Bias state variable at time2.
                 * @param options    Struct containing additional cost term options.
                 */
                explicit GyroSuperCostTerm(const Interface::ConstPtr &interface,
                                           const Time &time1, const Time &time2,
                                           const slam::eval::Evaluable<BiasType>::ConstPtr &bias1,
                                           const slam::eval::Evaluable<BiasType>::ConstPtr &bias2,
                                           const Options &options);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes the cost contribution to the objective function.
                 *
                 * Evaluates the difference between the estimated **gyro bias** and its prior,
                 * applying a **robust loss function** to mitigate outliers.
                 *
                 * @return The computed cost value.
                 */
                [[nodiscard]] double cost() const noexcept override;
                
                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves the set of related variable keys.
                 *
                 * This function returns the keys of state variables that influence the cost term.
                 *
                 * @param keys  Output container to store related variable keys.
                 */
                void getRelatedVarKeys(KeySet& keys) const noexcept override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Initializes precomputed interpolation matrices and Jacobians.
                 *
                 * This function precomputes necessary matrices to improve computational efficiency
                 * when evaluating the cost function.
                 */
                void init() { initialize_interp_matrices_(); }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Appends IMU data for cost term evaluation.
                 *
                 * Adds gyroscope bias observations over time, which are used to refine the
                 * bias estimation through optimization.
                 *
                 * @param imu_data  IMU data containing gyroscope bias readings.
                 */
                void emplace_back(const IMUData &imu_data) { imu_data_vec_.emplace_back(imu_data); }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Clears stored IMU data.
                 *
                 * This function resets the IMU measurement buffer.
                 */
                void clear() { imu_data_vec_.clear(); }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Reserves space for IMU data storage.
                 *
                 * This function preallocates memory to optimize storage of IMU measurements,
                 * reducing the need for dynamic memory allocations.
                 *
                 * @param N  Number of IMU measurements expected.
                 */
                void reserve(unsigned int N) { imu_data_vec_.reserve(N); }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves stored IMU data.
                 *
                 * @return Reference to the vector of stored IMU data.
                 */
                const std::vector<IMUData>& get() const { return imu_data_vec_; }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Sets IMU data for processing.
                 *
                 * @param imu_data_vec  Vector containing IMU measurements.
                 */
                void set(const std::vector<IMUData>& imu_data_vec) { imu_data_vec_ = imu_data_vec; }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes and accumulates Gauss-Newton terms for optimization.
                 *
                 * This function contributes the **Hessian matrix** (left-hand side) and
                 * **gradient vector** (right-hand side) used in solving the optimization problem.
                 *
                 * @param state_vec            Current state vector.
                 * @param approximate_hessian  Pointer to the Hessian approximation matrix.
                 * @param gradient_vector      Pointer to the gradient vector.
                 */
                void buildGaussNewtonTerms(const StateVector &state_vec,
                                          slam::blockmatrix::BlockSparseMatrix *approximate_hessian,
                                          slam::blockmatrix::BlockVector *gradient_vector) const override;

            private:

                const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

                // -----------------------------------------------------------------------------
                /** @brief Pointer to the trajectory interface containing state variables.
                 *
                 * This stores the **trajectory representation**, which provides access to the 
                 * system's pose, velocity, and acceleration estimates over time.
                 */
                const Interface::ConstPtr interface_;

                // -----------------------------------------------------------------------------
                /** @brief Start and end times for gyroscope and accelerometer bias estimation.
                 *
                 * These timestamps define the interval over which IMU biases are estimated.
                 */
                const Time time1_, time2_;

                // -----------------------------------------------------------------------------
                /** @brief Bias state variables at `time1_` and `time2_`.
                 *
                 * These represent the estimated gyroscope and accelerometer biases
                 * at two different points in time.
                 */
                const slam::eval::Evaluable<BiasType>::ConstPtr bias1_, bias2_;

                // -----------------------------------------------------------------------------
                /** @brief Configuration options for loss functions and bias estimation settings. */
                const Options options_;

                // -----------------------------------------------------------------------------
                /** @brief Trajectory knots corresponding to `time1_` and `time2_`.
                 *
                 * These trajectory states store the pose, velocity, and acceleration
                 * at the given timestamps, which are used for bias estimation.
                 */
                const Variable::ConstPtr knot1_, knot2_;

                // -----------------------------------------------------------------------------
                /** @brief Precomputed inverse covariance matrix for process noise.
                 *
                 * Stores \( Q^{-1}(T) \), the inverse process noise covariance used in
                 * state transition modeling.
                 */
                Matrix12d Qinv_T_ = Matrix12d::Identity();

                // -----------------------------------------------------------------------------
                /** @brief Precomputed state transition matrix.
                 *
                 * Stores the state transition matrix \( \Phi(T) \), which propagates 
                 * states over the time interval \( T \).
                 */
                Matrix12d Tran_T_ = Matrix12d::Identity();

                // -----------------------------------------------------------------------------
                /** @brief Precomputed interpolation matrices for IMU data alignment.
                 *
                 * Maps timestamps to rotation matrices for efficient pose and velocity interpolation:
                 * - First matrix: Transformation from IMU to world frame.
                 * - Second matrix: Transformation from world frame to IMU.
                 * - Uses `tbb::concurrent_hash_map` for thread-safe parallel access.
                 */
                tbb::concurrent_hash_map<double, std::pair<Eigen::Matrix4d, Eigen::Matrix4d>> interp_mats_;

                // -----------------------------------------------------------------------------
                /** @brief Container storing raw IMU data measurements.
                 *
                 * This vector accumulates **gyroscope readings over time**, 
                 * which are used to model bias drift in the optimization problem.
                 */
                std::vector<IMUData> imu_data_vec_;

                // -----------------------------------------------------------------------------
                /** @brief Stores measurement timestamps for IMU readings.
                 *
                 * These timestamps correspond to when gyroscope measurements were recorded.
                 */
                std::vector<double> meas_times_;

                // -----------------------------------------------------------------------------
                /** @brief Loss function for robust gyroscope bias optimization.
                 *
                 * This function applies a robust **M-estimator loss function** (e.g., Cauchy, DCS)
                 * to minimize the impact of outliers in gyroscope measurements.
                 */
                slam::problem::lossfunc::BaseLossFunc::Ptr gyro_loss_func_;

                // -----------------------------------------------------------------------------
                /** @brief Noise model for gyroscope bias estimation.
                 *
                 * Represents the **static noise covariance** associated with gyroscope bias errors.
                 * This is used in the optimization process to **weight error contributions**.
                 */
                slam::problem::noisemodel::StaticNoiseModel<3>::Ptr gyro_noise_model_ = slam::problem::noisemodel::StaticNoiseModel<3>::MakeShared(R);

                // -----------------------------------------------------------------------------
                /** @brief Jacobian matrix for velocity propagation.
                 *
                 * This **\( 3 \times 6 \) Jacobian** maps gyroscope measurements to velocity updates.
                 */
                Eigen::Matrix<double, 3, 6> jac_vel_ = Eigen::Matrix<double, 3, 6>::Zero();

                // -----------------------------------------------------------------------------
                /** @brief Jacobian matrix for gyroscope bias propagation.
                 *
                 * This **\( 3 \times 6 \) Jacobian** maps gyroscope bias measurements to error terms.
                 */
                Eigen::Matrix<double, 3, 6> jac_bias_gyro_ = Eigen::Matrix<double, 3, 6>::Zero();

                // -----------------------------------------------------------------------------
                /** @brief Initializes precomputed interpolation matrices.
                 *
                 * This function **precomputes** rotation and transformation matrices for efficient 
                 * interpolation of IMU bias measurements over time.
                 */
                void initialize_interp_matrices_();

            };

        }  // namespace costterm
    }  // namespace problem
}  // namespace slam
