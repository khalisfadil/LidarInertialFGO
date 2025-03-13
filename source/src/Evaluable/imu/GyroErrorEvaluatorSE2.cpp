#include "Evaluable/imu/GyroErrorEvaluatorSE2.hpp"

namespace slam {
    namespace eval {
        namespace imu {
            
            // -----------------------------------------------------------------------------
            // @brief Factory method to create an instance.
            // -----------------------------------------------------------------------------

            auto GyroErrorEvaluatorSE2::MakeShared(const Evaluable<VelInType>::ConstPtr &velocity,
                                                const Evaluable<BiasInType>::ConstPtr &bias,
                                                const ImuInType &gyro_meas) -> Ptr {
                return std::make_shared<GyroErrorEvaluatorSE2>(velocity, bias, gyro_meas);
            }

            // -----------------------------------------------------------------------------
            // @brief Constructor for GyroErrorEvaluatorSE2.
            // -----------------------------------------------------------------------------

            GyroErrorEvaluatorSE2::GyroErrorEvaluatorSE2(const Evaluable<VelInType>::ConstPtr &velocity,
                                                        const Evaluable<BiasInType>::ConstPtr &bias,
                                                        const ImuInType &gyro_meas)
                : velocity_(velocity), bias_(bias), gyro_meas_(gyro_meas) {
                jac_vel_.setZero();
                jac_bias_.setZero();
                jac_vel_(0, 5) = 1;
                jac_bias_(0, 5) = -1;
            }

            // -----------------------------------------------------------------------------
            // @brief Checks if the evaluator is active.
            // -----------------------------------------------------------------------------
            
            bool GyroErrorEvaluatorSE2::active() const {
                return velocity_->active() || bias_->active();
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Collects state variable keys that influence this evaluator.
             */
            void GyroErrorEvaluatorSE2::getRelatedVarKeys(KeySet &keys) const {
                velocity_->getRelatedVarKeys(keys);
                bias_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // @brief Computes the gyroscope error.
            // -----------------------------------------------------------------------------

            auto GyroErrorEvaluatorSE2::value() const -> OutType {
                return OutType(gyro_meas_(2, 0) + velocity_->value()(5, 0) - bias_->value()(5, 0));
            }

            // -----------------------------------------------------------------------------
            // @brief Forward evaluation of gyroscope error.
            // -----------------------------------------------------------------------------

            auto GyroErrorEvaluatorSE2::forward() const -> Node<OutType>::Ptr {
                const auto child1 = velocity_->forward();
                const auto child2 = bias_->forward();

                auto node = Node<OutType>::MakeShared(gyro_meas_(2, 0) + child1->value()(5, 0) - child2->value()(5, 0));
                node->addChild(child1);
                node->addChild(child2);

                return node;
            }


            // -----------------------------------------------------------------------------
            // @brief Computes Jacobians for the gyroscope error.
            // -----------------------------------------------------------------------------
            
            void GyroErrorEvaluatorSE2::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                                const Node<OutType>::Ptr& node,
                                                StateKeyJacobians& jacs) const {
                const auto child1 = std::static_pointer_cast<Node<VelInType>>(node->getChild(0));
                const auto child2 = std::static_pointer_cast<Node<BiasInType>>(node->getChild(1));

                if (velocity_->active()) {
                    velocity_->backward(lhs * jac_vel_, child1, jacs);
                }
                if (bias_->active()) {
                    bias_->backward(lhs * jac_bias_, child2, jacs);
                }
            }

            // -----------------------------------------------------------------------------
            // @brief Factory function for creating a GyroErrorEvaluatorSE2.
            // -----------------------------------------------------------------------------

            auto GyroErrorSE2(const Evaluable<GyroErrorEvaluatorSE2::VelInType>::ConstPtr &velocity,
                            const Evaluable<GyroErrorEvaluatorSE2::BiasInType>::ConstPtr &bias,
                            const GyroErrorEvaluatorSE2::ImuInType &gyro_meas) -> GyroErrorEvaluatorSE2::Ptr {
                return GyroErrorEvaluatorSE2::MakeShared(velocity, bias, gyro_meas);
            }

        }  // namespace imu
    }  // namespace eval
}  // namespace slam
