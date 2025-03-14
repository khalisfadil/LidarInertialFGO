#pragma once

#include <Eigen/Core>
#include "LGMath/LieGroupMath.hpp"
#include "Core/Evaluable/Evaluable.hpp"
#include "Core/Trajectory/Time.hpp"

namespace slam {
    namespace eval {
        namespace imu {

            // -----------------------------------------------------------------------------
            /**
             * @class GyroErrorEvaluator
             * @brief Evaluates gyroscope error for factor graph optimization.
             *
             * Computes gyroscope measurement error given an estimated velocity and bias state.
             * Used in IMU-based constraints for SLAM and sensor fusion.
             */
            class GyroErrorEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
            public:
                using Ptr = std::shared_ptr<GyroErrorEvaluator>;
                using ConstPtr = std::shared_ptr<const GyroErrorEvaluator>;
                using VelInType = Eigen::Matrix<double, 6, 1>;
                using BiasInType = Eigen::Matrix<double, 6, 1>;
                using ImuInType = Eigen::Matrix<double, 3, 1>;
                using OutType = Eigen::Matrix<double, 3, 1>;
                using Time = slam::traj::Time;
                using JacType = Eigen::Matrix<double, 3, 6>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create an instance.
                 * @param velocity Estimated velocity state.
                 * @param bias Estimated IMU bias.
                 * @param gyro_meas Measured gyroscope data.
                 * @return Shared pointer to a new evaluator instance.
                 */
                static Ptr MakeShared(const Evaluable<VelInType>::ConstPtr &velocity,
                                    const Evaluable<BiasInType>::ConstPtr &bias,
                                    const ImuInType &gyro_meas);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor.
                 * @param velocity Estimated velocity state.
                 * @param bias Estimated IMU bias.
                 * @param gyro_meas Measured gyroscope data.
                 */
                GyroErrorEvaluator(const Evaluable<VelInType>::ConstPtr &velocity,
                                const Evaluable<BiasInType>::ConstPtr &bias,
                                const ImuInType &gyro_meas);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Checks if the gyroscope error is influenced by active state variables.
                 * @return True if active, otherwise false.
                 */
                bool active() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Collects state variable keys that influence this evaluator.
                 * @param[out] keys Set of related state keys.
                 */
                void getRelatedVarKeys(KeySet& keys) const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes the gyroscope measurement error.
                 * @return Gyroscope error as a 3x1 matrix.
                 */
                OutType value() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Forward evaluation of gyroscope measurement error.
                 * @return Shared pointer to the computed gyroscope error node.
                 */
                Node<OutType>::Ptr forward() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes Jacobians for the gyroscope measurement error.
                 * @param lhs Left-hand-side weight matrix.
                 * @param node Velocity state node.
                 * @param jacs Container for accumulating Jacobians.
                 */
                void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                            const Node<OutType>::Ptr& node,
                            StateKeyJacobians& jacs) const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Sets the timestamp for the gyroscope measurement.
                 * @param time Timestamp.
                 */
                void setTime(Time time) { 
                    time_ = time; 
                    time_init_ = true; 
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves the timestamp for the gyroscope measurement.
                 * @return Timestamp.
                 * @throws std::runtime_error if the timestamp was not initialized.
                 */
                Time getTime() const {
                    return time_init_ ? time_ : throw std::runtime_error("[GyroErrorEvaluator::getTime] Time was not initialized");
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes measurement Jacobians.
                 * @param[out] jac_vel Jacobian w.r.t. velocity.
                 * @param[out] jac_bias Jacobian w.r.t. bias.
                 */
                void getMeasJacobians(JacType &jac_vel, JacType &jac_bias) const;

            private:
                const Evaluable<VelInType>::ConstPtr velocity_;  ///< Estimated velocity state.
                const Evaluable<BiasInType>::ConstPtr bias_;    ///< Estimated IMU bias.
                const ImuInType gyro_meas_;                     ///< Measured gyroscope data.

                JacType jac_vel_ = JacType::Zero();             ///< Velocity Jacobian.
                JacType jac_bias_ = JacType::Zero();            ///< Bias Jacobian.

                Time time_;                                     ///< Timestamp of measurement.
                bool time_init_ = false;                        ///< Flag indicating if time was set.
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory function for creating a GyroErrorEvaluator.
             * @param velocity Estimated velocity state.
             * @param bias Estimated IMU bias.
             * @param gyro_meas Measured gyroscope data.
             * @return Shared pointer to the evaluator.
             */
            auto GyroError(const Evaluable<GyroErrorEvaluator::VelInType>::ConstPtr &velocity,
                        const Evaluable<GyroErrorEvaluator::BiasInType>::ConstPtr &bias,
                        const GyroErrorEvaluator::ImuInType &gyro_meas) -> GyroErrorEvaluator::Ptr;

        }  // namespace imu
    }  // namespace eval
}  // namespace slam
