#pragma once

#include <Eigen/Core>
#include "LGMath/LieGroupMath.hpp"
#include "Core/Evaluable/Evaluable.hpp"

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
            class IMUErrorEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
            public:
                using Ptr = std::shared_ptr<IMUErrorEvaluator>;
                using ConstPtr = std::shared_ptr<const IMUErrorEvaluator>;

                using PoseInType = liemath::se3::Transformation;
                using VelInType = Eigen::Matrix<double, 6, 1>;
                using AccInType = Eigen::Matrix<double, 6, 1>;
                using BiasInType = Eigen::Matrix<double, 6, 1>;
                using ImuInType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 6, 1>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create an instance.
                 * @param velocity Estimated velocity state.
                 * @param bias Estimated IMU bias.
                 * @param gyro_meas Measured gyroscope data.
                 * @return Shared pointer to a new evaluator instance.
                 */
                static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr &transform,
                                        const Evaluable<VelInType>::ConstPtr &velocity,
                                        const Evaluable<AccInType>::ConstPtr &acceleration,
                                        const Evaluable<BiasInType>::ConstPtr &bias,
                                        const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                                        const ImuInType &imu_meas);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor.
                 * @param velocity Estimated velocity state.
                 * @param bias Estimated IMU bias.
                 * @param gyro_meas Measured gyroscope data.
                 */
                IMUErrorEvaluator(const Evaluable<PoseInType>::ConstPtr &transform,
                                    const Evaluable<VelInType>::ConstPtr &velocity,
                                    const Evaluable<AccInType>::ConstPtr &acceleration,
                                    const Evaluable<BiasInType>::ConstPtr &bias,
                                    const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                                    const ImuInType &imu_meas);

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
                 * @brief Sets the gravity vector.
                 * @param gravity Gravity magnitude along Z-axis.
                 */
                void setGravity(double gravity) { gravity_(2, 0) = gravity; }

            private:
                // evaluable
                const Evaluable<PoseInType>::ConstPtr transform_;
                const Evaluable<VelInType>::ConstPtr velocity_;
                const Evaluable<AccInType>::ConstPtr acceleration_;
                const Evaluable<BiasInType>::ConstPtr bias_;
                const Evaluable<PoseInType>::ConstPtr transform_i_to_m_;
                const ImuInType imu_meas_;
                Eigen::Matrix<double, 3, 1> gravity_ = Eigen::Matrix<double, 3, 1>::Zero();
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory function for creating a GyroErrorEvaluator.
             * @param velocity Estimated velocity state.
             * @param bias Estimated IMU bias.
             * @param gyro_meas Measured gyroscope data.
             * @return Shared pointer to the evaluator.
             */
            auto IMUErrorEvaluator(const Evaluable<IMUErrorEvaluator::PoseInType>::ConstPtr &transform,
                                    const Evaluable<IMUErrorEvaluator::VelInType>::ConstPtr &velocity,
                                    const Evaluable<IMUErrorEvaluator::AccInType>::ConstPtr &acceleration,
                                    const Evaluable<IMUErrorEvaluator::BiasInType>::ConstPtr &bias,
                                    const Evaluable<IMUErrorEvaluator::PoseInType>::ConstPtr &transform_i_to_m,
                                    const IMUErrorEvaluator::ImuInType &imu_meas) -> IMUErrorEvaluator::Ptr;

        }  // namespace imu
    }  // namespace eval
}  // namespace slam
