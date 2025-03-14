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
             * @class AccelerationErrorEvaluator
             * @brief Evaluates acceleration error for factor graph optimization.
             *
             * Computes acceleration error given a measured acceleration and an estimated acceleration state.
             * Used in IMU-based constraints for SLAM and sensor fusion.
             */
            class AccelerationErrorEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
            public:
                using Ptr = std::shared_ptr<AccelerationErrorEvaluator>;
                using ConstPtr = std::shared_ptr<const AccelerationErrorEvaluator>;
                using PoseInType = liemath::se3::Transformation;
                using AccInType = Eigen::Matrix<double, 6, 1>;
                using BiasInType = Eigen::Matrix<double, 6, 1>;
                using ImuInType = Eigen::Matrix<double, 3, 1>;
                using OutType = Eigen::Matrix<double, 3, 1>;
                using Time = slam::traj::Time;
                using JacType = Eigen::Matrix<double, 3, 6>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create an instance.
                 * @param transform Evaluated SE(3) transformation.
                 * @param acceleration Estimated acceleration state.
                 * @param bias IMU bias.
                 * @param transform_i_to_m Transformation from IMU to measurement frame.
                 * @param acc_meas Measured acceleration.
                 * @return Shared pointer to a new evaluator instance.
                 */
                static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr &transform,
                                    const Evaluable<AccInType>::ConstPtr &acceleration,
                                    const Evaluable<BiasInType>::ConstPtr &bias,
                                    const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                                    const ImuInType &acc_meas);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor.
                 * @param transform Evaluated SE(3) transformation.
                 * @param acceleration Estimated acceleration state.
                 * @param bias IMU bias.
                 * @param transform_i_to_m Transformation from IMU to measurement frame.
                 * @param acc_meas Measured acceleration.
                 */
                AccelerationErrorEvaluator(const Evaluable<PoseInType>::ConstPtr &transform,
                                        const Evaluable<AccInType>::ConstPtr &acceleration,
                                        const Evaluable<BiasInType>::ConstPtr &bias,
                                        const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                                        const ImuInType &acc_meas);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Checks if the acceleration error is influenced by active state variables.
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
                 * @brief Computes the acceleration error.
                 * @return Acceleration error as a 3x1 matrix.
                 */
                OutType value() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Forward evaluation of acceleration error.
                 * @return Shared pointer to the computed acceleration error node.
                 */
                Node<OutType>::Ptr forward() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes Jacobians for the acceleration error.
                 * @param lhs Left-hand-side weight matrix.
                 * @param node Acceleration state node.
                 * @param jacs Container for accumulating Jacobians.
                 */
                void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                            const Node<OutType>::Ptr& node,
                            StateKeyJacobians& jacs) const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Sets the gravity vector.
                 * @param gravity Gravity magnitude along Z-axis.
                 */
                void setGravity(double gravity) { gravity_(2, 0) = gravity; }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Sets the timestamp for the acceleration measurement.
                 * @param time Timestamp.
                 */
                void setTime(Time time) { 
                    time_ = time; 
                    time_init_ = true; 
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves the timestamp for the acceleration measurement.
                 * @return Timestamp.
                 * @throws std::runtime_error if the timestamp was not initialized.
                 */
                Time getTime() const {
                    return time_init_ ? time_ : throw std::runtime_error("[AccelerationErrorEvaluator::getTime] Time was not initialized");
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes measurement Jacobians.
                 * @param[out] jac_pose Jacobian w.r.t. pose.
                 * @param[out] jac_accel Jacobian w.r.t. acceleration.
                 * @param[out] jac_bias Jacobian w.r.t. bias.
                 * @param[out] jac_T_mi Jacobian w.r.t. IMU-to-measurement transformation.
                 */
                void getMeasJacobians(JacType &jac_pose, JacType &jac_accel,
                                    JacType &jac_bias, JacType &jac_T_mi) const;

            private:
                const Evaluable<PoseInType>::ConstPtr transform_;          ///< Transformation state.
                const Evaluable<AccInType>::ConstPtr acceleration_;        ///< Acceleration state.
                const Evaluable<BiasInType>::ConstPtr bias_;              ///< IMU bias.
                const Evaluable<PoseInType>::ConstPtr transform_i_to_m_;  ///< IMU-to-measurement transformation.
                const ImuInType acc_meas_;                                ///< Measured acceleration.

                JacType jac_accel_ = JacType::Zero();                     ///< Acceleration Jacobian.
                JacType jac_bias_ = JacType::Zero();                      ///< Bias Jacobian.
                Eigen::Matrix<double, 3, 1> gravity_ = Eigen::Matrix<double, 3, 1>::Zero(); ///< Gravity vector.
                Time time_;                                               ///< Timestamp of measurement.
                bool time_init_ = false;                                  ///< Flag indicating if time was set.
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory function for creating an AccelerationErrorEvaluator.
             * @param transform Evaluated SE(3) transformation.
             * @param acceleration Estimated acceleration state.
             * @param bias IMU bias.
             * @param transform_i_to_m Transformation from IMU to measurement frame.
             * @param acc_meas Measured acceleration.
             * @return Shared pointer to the evaluator.
             */
            auto AccelerationError(const Evaluable<AccelerationErrorEvaluator::PoseInType>::ConstPtr &transform,
                                                        const Evaluable<AccelerationErrorEvaluator::AccInType>::ConstPtr &acceleration,
                                                        const Evaluable<AccelerationErrorEvaluator::BiasInType>::ConstPtr &bias,
                                                        const Evaluable<AccelerationErrorEvaluator::PoseInType>::ConstPtr &transform_i_to_m,
                                                        const AccelerationErrorEvaluator::ImuInType &acc_meas) -> AccelerationErrorEvaluator::Ptr;

        }  // namespace imu
    }  // namespace eval
}  // namespace slam
