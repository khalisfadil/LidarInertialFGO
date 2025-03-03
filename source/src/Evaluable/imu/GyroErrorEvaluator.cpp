#include "source/include/Evaluable/imu/GyroErrorEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace imu {
            
            // -----------------------------------------------------------------------------
            // @brief Factory method to create an instance.
            // -----------------------------------------------------------------------------

            auto GyroErrorEvaluator::MakeShared(const Evaluable<VelInType>::ConstPtr &velocity,
                                                const Evaluable<BiasInType>::ConstPtr &bias,
                                                const ImuInType &gyro_meas) -> Ptr {
                return std::make_shared<GyroErrorEvaluator>(velocity, bias, gyro_meas);
            }

            // -----------------------------------------------------------------------------
            // @brief Constructor for GyroErrorEvaluatorSE2.
            // -----------------------------------------------------------------------------

            GyroErrorEvaluator::GyroErrorEvaluator(
                const Evaluable<VelInType>::ConstPtr &velocity,
                const Evaluable<BiasInType>::ConstPtr &bias,
                const ImuInType &gyro_meas)
                : velocity_(velocity), bias_(bias), gyro_meas_(gyro_meas) {
                jac_vel_.setZero();
                jac_bias_.setZero();
                jac_vel_.rightCols<3>().setIdentity();
                jac_bias_.rightCols<3>().setIdentity().array() *= -1;
            }


            // -----------------------------------------------------------------------------
            // @brief Checks if the evaluator is active.
            // -----------------------------------------------------------------------------
            
            bool GyroErrorEvaluator::active() const {
                return velocity_->active() || bias_->active();
            }

            // -----------------------------------------------------------------------------
            // @brief Collects state variable keys that influence this evaluator.
            // -----------------------------------------------------------------------------
            
            void GyroErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
                velocity_->getRelatedVarKeys(keys);
                bias_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // @brief Computes the gyroscope error.
            // -----------------------------------------------------------------------------

            auto GyroErrorEvaluator::value() const -> OutType {
                return gyro_meas_ + velocity_->value().tail<3>() - bias_->value().tail<3>();
            }

            // -----------------------------------------------------------------------------
            // @brief Forward evaluation of gyroscope error.
            // -----------------------------------------------------------------------------

            auto GyroErrorEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child1 = velocity_->forward();
                const auto child2 = bias_->forward();

                auto node = Node<OutType>::MakeShared(gyro_meas_ + child1->value().tail<3>() - child2->value().tail<3>());
                node->addChild(child1);
                node->addChild(child2);

                return node;
            }

            // -----------------------------------------------------------------------------
            // @brief Computes Jacobians for the gyroscope error.
            // -----------------------------------------------------------------------------

            void GyroErrorEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
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

            auto GyroError(const Evaluable<GyroErrorEvaluator::VelInType>::ConstPtr &velocity,
                                const Evaluable<GyroErrorEvaluator::BiasInType>::ConstPtr &bias,
                                const GyroErrorEvaluator::ImuInType &gyro_meas) -> GyroErrorEvaluator::Ptr {
                return GyroErrorEvaluator::MakeShared(velocity, bias, gyro_meas);
            }

        }  // namespace imu
    }  // namespace eval
}  // namespace slam
