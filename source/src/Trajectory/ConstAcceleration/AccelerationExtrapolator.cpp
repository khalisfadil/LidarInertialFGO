#include "source/include/Trajectory/ConstAcceleration/AccelerationExtrapolator.hpp"
#include "source/include/Trajectory/ConstAcceleration/Helper.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {
            
            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto AccelerationExtrapolator::MakeShared(const Time& time,
                                          const Variable::ConstPtr& knot) -> Ptr {
                return std::make_shared<AccelerationExtrapolator>(time, knot);
            }

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            AccelerationExtrapolator::AccelerationExtrapolator(const Time& time,
                                                   const Variable::ConstPtr& knot)
            : knot_(knot), Phi_(getTran((time - knot->getTime()).seconds())) {}

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto AccelerationExtrapolator::active() const -> bool {
                return knot_->getAcceleration()->active();
            }

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            void AccelerationExtrapolator::getRelatedVarKeys(KeySet& keys) const {
                knot_->getAcceleration()->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto AccelerationExtrapolator::value() const -> OutType {
                return OutType(Phi_.block<6, 6>(12, 12) * knot_->getAcceleration()->value());
            }

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto AccelerationExtrapolator::forward() const ->  slam::eval::Node<OutType>::Ptr {
                const auto dw = knot_->getAcceleration()->forward();

                // Compute interpolated acceleration and create node
                const auto node =  slam::eval::Node<OutType>::MakeShared(Phi_.block<6, 6>(12, 12) * dw->value());

                // Add child node
                node->addChild(dw);
                
                return node;
            }

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            void AccelerationExtrapolator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                        const slam::eval::Node<OutType>::Ptr& node,
                                        slam::eval::StateKeyJacobians& jacs) const {
                if (!active() || !knot_->getAcceleration()->active()) return;

                // Compute Jacobian transformation
                knot_->getAcceleration()->backward(lhs * Phi_.block<6, 6>(12, 12), 
                    std::static_pointer_cast<slam::eval::Node<InAccType>>(node->getChild(0)), jacs);
            }
            
        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
