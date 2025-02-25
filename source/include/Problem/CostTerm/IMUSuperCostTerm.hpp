#pragma once

#include <memory>
#include <vector>
#include <stdexcept>
#include <Eigen/Core>

#include <tbb/parallel_reduce.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/parallel_for.h>
#include <tbb/mutex.h>

#include "source/include/Evaluable/se3/Se3StateVariable.hpp"
#include "source/include/Evaluable/StateVariable.hpp"
#include "source/include/Evaluable/vspace/VSpaceStateVar.hpp"
#include "source/include/Problem/CostTerm/BaseCostTerm.hpp"
#include "source/include/Problem/LossFunc/LossFunc.hpp"
#include "source/include/Problem/NoiseModel/StaticNoiseModel.hpp"
#include "source/include/Problem/Problem.hpp"
#include "source/include/Trajectory/ConstAcceleration/Interface.hpp"
#include "source/include/Trajectory/Time.hpp"

namespace slam {
    namespace problem {
        namespace costterm {

            // -----------------------------------------------------------------------------
            /**
             * @struct IMUData
             * @brief Holds raw IMU measurement data.
             *
             * This structure represents a single IMU measurement containing:
             * - **timestamp**: IMU reading time.
             * - **ang_vel**: Angular velocity in the IMU frame (3D vector).
             * - **lin_acc**: Linear acceleration in the IMU frame (3D vector).
             */
            struct IMUData {
                double timestamp = 0.0;                   ///< Measurement timestamp (seconds).
                Eigen::Vector3d ang_vel = Eigen::Vector3d::Zero(); ///< Angular velocity (rad/s).
                Eigen::Vector3d lin_acc = Eigen::Vector3d::Zero(); ///< Linear acceleration (m/s²).

                IMUData() = default;
                IMUData(double timestamp_, Eigen::Vector3d ang_vel_, Eigen::Vector3d lin_acc_)
                    : timestamp(timestamp_), ang_vel(std::move(ang_vel_)), lin_acc(std::move(lin_acc_)) {}
            };

            // -----------------------------------------------------------------------------
            /**
             * @class IMUSuperCostTerm
             * @brief Implements an IMU-based cost term for factor graph optimization.
             *
             * This cost term integrates IMU constraints into optimization using:
             * - **IMU kinematic constraints** (accelerations, angular velocities).
             * - **Factor graph optimization** (relates IMU to trajectory states).
             * - **Gaussian noise models** for measurement uncertainty.
             *
             * **Key Responsibilities:**
             * - Computes cost contribution from IMU measurements.
             * - Tracks dependencies on state variables (pose, velocity, acceleration, bias).
             * - Builds Gauss-Newton system terms for optimization.
             */
            class IMUSuperCostTerm : public BaseCostTerm {
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
                 * @brief Stores configuration parameters for IMU-based optimization.
                 */
                struct Options {
                    LOSS_FUNC acc_loss_func = LOSS_FUNC::CAUCHY; ///< Loss function for acceleration residuals.
                    LOSS_FUNC gyro_loss_func = LOSS_FUNC::CAUCHY; ///< Loss function for gyro residuals.
                    double acc_loss_sigma = 0.1;       ///< Scale factor for acceleration loss function.
                    double gyro_loss_sigma = 0.1;      ///< Scale factor for gyroscope loss function.
                    Eigen::Vector3d gravity = Eigen::Vector3d::Zero(); ///< Gravity vector (m/s²).
                    Eigen::Vector3d r_imu_acc = Eigen::Vector3d::Zero(); ///< IMU accelerometer position (lever arm).
                    Eigen::Vector3d r_imu_ang = Eigen::Vector3d::Zero(); ///< IMU gyroscope position (lever arm).
                    bool se2 = false;                  ///< If true, enforces SE(2) motion constraints.
                    bool use_accel = true;             ///< If true, includes accelerometer constraints.
                };

                using Ptr = std::shared_ptr<IMUSuperCostTerm>; ///< Shared pointer type.
                using ConstPtr = std::shared_ptr<const IMUSuperCostTerm>;

                using PoseType = liemath::se3::Transformation; ///< SE(3) Pose.
                using VelType = Eigen::Matrix<double, 6, 1>;   ///< SE(3) Velocity (twist).
                using AccType = Eigen::Matrix<double, 6, 1>;   ///< SE(3) Acceleration.
                using BiasType = Eigen::Matrix<double, 6, 1>;  ///< Bias vector (gyro + accel).
                using Interface = slam::traj::const_acc::Interface;
                using Variable = slam::traj::const_acc::Variable;
                using Time = slam::traj::Time;
                using Matrix18d = Eigen::Matrix<double, 18, 18>;
                using Matrix6d = Eigen::Matrix<double, 6, 6>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create an instance of `IMUSuperCostTerm`.
                 *
                 * This method constructs an IMU-based cost term for trajectory optimization.
                 * It facilitates **bias estimation** and **error modeling** by leveraging sensor measurements.
                 * The method is preferred over direct instantiation to enable **shared ownership**.
                 *
                 * @param interface        Pointer to the **trajectory interface** containing system states.
                 * @param time1            Start time for IMU bias estimation.
                 * @param time2            End time for IMU bias estimation.
                 * @param bias1            Bias state variable at `time1`.
                 * @param bias2            Bias state variable at `time2`.
                 * @param transform_i_to_m_1 IMU-to-map transformation at `time1`.
                 * @param transform_i_to_m_2 IMU-to-map transformation at `time2`.
                 * @param options          Struct containing additional cost term options (e.g., loss function settings).
                 *
                 * @return Shared pointer to the constructed IMU cost term.
                 */
                static Ptr MakeShared(const Interface::ConstPtr& interface, 
                                    const Time time1, const Time time2,
                                    const slam::eval::Evaluable<BiasType>::ConstPtr& bias1,
                                    const slam::eval::Evaluable<BiasType>::ConstPtr& bias2,
                                    const slam::eval::Evaluable<PoseType>::ConstPtr& transform_i_to_m_1,
                                    const slam::eval::Evaluable<PoseType>::ConstPtr& transform_i_to_m_2,
                                    const Options& options);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs the IMU-based cost term for trajectory optimization.
                 *
                 * This constructor initializes the cost term using the provided **system states**
                 * and **IMU bias estimates** from sensor readings. The term models measurement errors 
                 * from **gyroscope and accelerometer data**, ensuring accurate bias correction.
                 *
                 * @param interface        Pointer to the **trajectory interface** containing system states.
                 * @param time1            Start time for IMU bias estimation.
                 * @param time2            End time for IMU bias estimation.
                 * @param bias1            Bias state variable at `time1`.
                 * @param bias2            Bias state variable at `time2`.
                 * @param transform_i_to_m_1 IMU-to-map transformation at `time1`.
                 * @param transform_i_to_m_2 IMU-to-map transformation at `time2`.
                 * @param options          Struct containing additional cost term options (e.g., loss function settings).
                 */
                IMUSuperCostTerm(const Interface::ConstPtr& interface, 
                                const Time time1, const Time time2,
                                const slam::eval::Evaluable<BiasType>::ConstPtr& bias1,
                                const slam::eval::Evaluable<BiasType>::ConstPtr& bias2,
                                const slam::eval::Evaluable<PoseType>::ConstPtr& transform_i_to_m_1,
                                const slam::eval::Evaluable<PoseType>::ConstPtr& transform_i_to_m_2,
                                const Options& options);

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

                // -----------------------------------------------------------------------------
                /** @brief Shared pointer to the trajectory interface.
                 *
                 * Provides access to the **trajectory representation**, which contains
                 * the state variables (pose, velocity, acceleration) used for bias estimation.
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
                /** @brief Transformation matrices from IMU to world frame.
                 *
                 * These transformations define the **pose relationship** between the IMU
                 * and the world frame at `time1_` and `time2_`.
                 */
                const slam::eval::Evaluable<PoseType>::ConstPtr transform_i_to_m_1_, transform_i_to_m_2_;

                // -----------------------------------------------------------------------------
                /** @brief Configuration settings for loss functions, bias estimation, and sensor parameters. */
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
                 * Stores **\( Q^{-1}(T) \)**, the inverse process noise covariance matrix used
                 * in state transition modeling.
                 */
                Matrix18d Qinv_T_ = Matrix18d::Identity();

                // -----------------------------------------------------------------------------
                /** @brief Precomputed state transition matrix.
                 *
                 * Stores the **transition matrix \( \Phi(T) \)**, which propagates
                 * states over the time interval \( T \).
                 */
                Matrix18d Tran_T_ = Matrix18d::Identity();

                // -----------------------------------------------------------------------------
                /** @brief Precomputed interpolation matrices for IMU data alignment.
                 *
                 * Maps timestamps to rotation matrices for efficient pose and velocity interpolation:
                 * - First matrix: Transformation from IMU to world frame.
                 * - Second matrix: Transformation from world frame to IMU.
                 * - Uses `tbb::concurrent_hash_map` for thread-safe parallel access.
                 */
                tbb::concurrent_hash_map<double, std::pair<Eigen::Matrix3d, Eigen::Matrix3d>> interp_mats_;


                // -----------------------------------------------------------------------------
                /** @brief Stores raw IMU data (gyroscope and accelerometer readings).
                 *
                 * This vector accumulates **IMU measurements over time**, which are used
                 * to model sensor bias and drift in optimization.
                 */
                std::vector<IMUData> imu_data_vec_;

                // -----------------------------------------------------------------------------
                /** @brief Stores measurement timestamps for IMU readings.
                 *
                 * These timestamps correspond to the recorded IMU measurements used in bias estimation.
                 */
                std::vector<double> meas_times_;

                // -----------------------------------------------------------------------------
                /** @brief Loss function for accelerometer bias estimation.
                 *
                 * This function applies a robust **M-estimator loss function** (e.g., Cauchy, DCS)
                 * to reduce the impact of outliers in accelerometer measurements.
                 */
                slam::problem::lossfunc::BaseLossFunc::Ptr acc_loss_func_ = slam::problem::lossfunc::L2LossFunc::MakeShared();

                // -----------------------------------------------------------------------------
                /** @brief Loss function for gyroscope bias estimation.
                 *
                 * Similar to `acc_loss_func_`, this function helps mitigate outlier effects
                 * in gyroscope bias estimation.
                 */
                slam::problem::lossfunc::BaseLossFunc::Ptr gyro_loss_func_ = slam::problem::lossfunc::L2LossFunc::MakeShared();

                // -----------------------------------------------------------------------------
                /** @brief Noise model for accelerometer bias estimation.
                 *
                 * Represents the **static noise covariance** for accelerometer bias errors,
                 * used in optimization to weight error contributions appropriately.
                 */
                slam::problem::noisemodel::StaticNoiseModel<3>::Ptr acc_noise_model_ = slam::problem::noisemodel::StaticNoiseModel<3>::MakeShared(R);

                // -----------------------------------------------------------------------------
                /** @brief Noise model for gyroscope bias estimation.
                 *
                 * Similar to `acc_noise_model_`, this defines the **static noise covariance**
                 * for gyroscope bias errors.
                 */
                slam::problem::noisemodel::StaticNoiseModel<3>::Ptr gyro_noise_model_ = slam::problem::noisemodel::StaticNoiseModel<3>::MakeShared(R);

                // -----------------------------------------------------------------------------
                /** @brief Jacobian matrix mapping velocity updates.
                 *
                 * This **\( 3 \times 6 \)** matrix relates gyroscope measurements to velocity changes.
                 */
                Eigen::Matrix<double, 3, 6> jac_vel_ = Eigen::Matrix<double, 3, 6>::Zero();

                // -----------------------------------------------------------------------------
                /** @brief Jacobian matrix mapping accelerometer updates.
                 *
                 * This **\( 3 \times 6 \)** matrix relates accelerometer measurements to acceleration changes.
                 */
                Eigen::Matrix<double, 3, 6> jac_accel_ = Eigen::Matrix<double, 3, 6>::Zero();

                // -----------------------------------------------------------------------------
                /** @brief Jacobian matrix mapping accelerometer bias to errors.
                 *
                 * This **\( 3 \times 6 \)** matrix models the effect of accelerometer bias
                 * on estimated acceleration.
                 */
                Eigen::Matrix<double, 3, 6> jac_bias_accel_ = Eigen::Matrix<double, 3, 6>::Zero();

                // -----------------------------------------------------------------------------
                /** @brief Jacobian matrix mapping gyroscope bias to errors.
                 *
                 * This **\( 3 \times 6 \)** matrix models the effect of gyroscope bias
                 * on estimated velocity and rotation.
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
