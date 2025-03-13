#include "Evaluable/imu/DmiErrorEvaluator.hpp"
#include "LGMath/LieGroupMath.hpp"

namespace slam {
    namespace eval {
        namespace imu {

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory method to create an instance.
             */
            auto DMIErrorEvaluator::MakeShared(const Evaluable<VelInType>::ConstPtr &velocity,
                                            const Evaluable<ScaleInType>::ConstPtr &scale,
                                            DMIInType dmi_meas) -> Ptr {
                return std::make_shared<DMIErrorEvaluator>(velocity, scale, dmi_meas);
            }

            // -----------------------------------------------------------------------------
            // @brief Factory function for creating a DMIErrorEvaluator.
            // -----------------------------------------------------------------------------
            
            DMIErrorEvaluator::DMIErrorEvaluator(const Evaluable<VelInType>::ConstPtr &velocity,
                                                const Evaluable<ScaleInType>::ConstPtr &scale,
                                                DMIInType dmi_meas)
                : velocity_(velocity), scale_(scale), dmi_meas_(dmi_meas) {
                jac_vel_.setIdentity();
                jac_scale_(0, 0) = dmi_meas_;
            }

            // -----------------------------------------------------------------------------
            // @brief Factory function for creating a DMIErrorEvaluator.
            // -----------------------------------------------------------------------------

            bool DMIErrorEvaluator::active() const {
                return velocity_->active() || scale_->active();
            }

            // -----------------------------------------------------------------------------
            // @brief Factory function for creating a DMIErrorEvaluator.
            // -----------------------------------------------------------------------------

            void DMIErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
                velocity_->getRelatedVarKeys(keys);
                scale_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // @brief Factory function for creating a DMIErrorEvaluator.
            // -----------------------------------------------------------------------------

            auto DMIErrorEvaluator::value() const -> OutType {
                return OutType(dmi_meas_ * scale_->value()(0, 0) + velocity_->value()(0, 0));
            }

            // -----------------------------------------------------------------------------
            // @brief Factory function for creating a DMIErrorEvaluator.
            // -----------------------------------------------------------------------------

            auto DMIErrorEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child1 = velocity_->forward();
                const auto child2 = scale_->forward();

                auto node = Node<OutType>::MakeShared(dmi_meas_ * child2->value()(0, 0) + child1->value()(0, 0));
                node->addChild(child1);
                node->addChild(child2);

                return node;
            }

            // -----------------------------------------------------------------------------
            // @brief Factory function for creating a DMIErrorEvaluator.
            // -----------------------------------------------------------------------------

            void DMIErrorEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                            const Node<OutType>::Ptr& node,
                                            StateKeyJacobians& jacs) const {
                const auto child1 = std::static_pointer_cast<Node<VelInType>>(node->getChild(0));
                const auto child2 = std::static_pointer_cast<Node<ScaleInType>>(node->getChild(1));

                if (velocity_->active()) {
                    velocity_->backward(lhs * jac_vel_, child1, jacs);
                }
                if (scale_->active()) {
                    scale_->backward(lhs * jac_scale_, child2, jacs);
                }
            }

            // -----------------------------------------------------------------------------
            // @brief Factory function for creating a DMIErrorEvaluator.
            // -----------------------------------------------------------------------------

            auto DMIError(const Evaluable<DMIErrorEvaluator::VelInType>::ConstPtr &velocity,
                        const Evaluable<DMIErrorEvaluator::ScaleInType>::ConstPtr &scale,
                        DMIErrorEvaluator::DMIInType dmi_meas) -> DMIErrorEvaluator::Ptr {
                return DMIErrorEvaluator::MakeShared(velocity, scale, dmi_meas);
            }

        }  // namespace imu
    }  // namespace eval
}  // namespace slam
