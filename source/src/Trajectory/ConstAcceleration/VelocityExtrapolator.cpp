#include "source/include/Trajectory/ConstAcceleration/VelocityExtrapolator.hpp"

#include "source/include/Trajectory/ConstAcceleration/Helper.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto VelocityExtrapolator::MakeShared(const Time time,
                                                const Variable::ConstPtr& knot) -> Ptr {
                return std::make_shared<VelocityExtrapolator>(time, knot);
            }

            // -----------------------------------------------------------------------------
            // Constructor
            // -----------------------------------------------------------------------------

            VelocityExtrapolator::VelocityExtrapolator(const Time time,
                                                    const Variable::ConstPtr& knot)
                : knot_(knot) {
                // Compute transition matrix for the given time shift
                const double tau = (time - knot->getTime()).seconds();
                Phi_ = getTran(tau);
            }

            // -----------------------------------------------------------------------------
            // Active Check
            // -----------------------------------------------------------------------------

            bool VelocityExtrapolator::active() const {
                return knot_->getVelocity()->active() || knot_->getAcceleration()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void VelocityExtrapolator::getRelatedVarKeys(eval::Evaluable<OutType>::KeySet& keys) const {
                knot_->getVelocity()->getRelatedVarKeys(keys);
                knot_->getAcceleration()->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto VelocityExtrapolator::value() const -> OutType {
                return Phi_.block<6, 6>(6, 6) * knot_->getVelocity()->value() +
                    Phi_.block<6, 6>(6, 12) * knot_->getAcceleration()->value();
            }


            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto VelocityExtrapolator::forward() const -> slam::eval::Node<OutType>::Ptr {
                const auto w = knot_->getVelocity()->forward(), dw = knot_->getAcceleration()->forward();

                // Compute interpolated velocity
                const auto node = slam::eval::Node<OutType>::MakeShared(
                    Phi_.block<6, 6>(6, 6) * w->value() + Phi_.block<6, 6>(6, 12) * dw->value()
                );

                // Explicitly specify the container type
                std::initializer_list<slam::eval::NodeBase::Ptr> children = {w, dw};
                for (const auto& child : children) node->addChild(child);

                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void VelocityExtrapolator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                    const eval::Node<OutType>::Ptr& node,
                                    eval::StateKeyJacobians& jacs) const {
                if (!active()) return;

                // Precompute Phi matrix blocks
                const auto phi_v = Phi_.block<6, 6>(6, 6);
                const auto phi_a = Phi_.block<6, 6>(6, 12);

                // Lambda-based Jacobian updates
                std::array<std::function<void()>, 2> jacobian_updates = {
                    [&] { if (knot_->getVelocity()->active()) 
                            knot_->getVelocity()->backward(lhs * phi_v, 
                                std::static_pointer_cast<eval::Node<InVelType>>(node->getChild(1)), jacs); },
                    [&] { if (knot_->getAcceleration()->active()) 
                            knot_->getAcceleration()->backward(lhs * phi_a, 
                                std::static_pointer_cast<eval::Node<InAccType>>(node->getChild(2)), jacs); }
                };

                for (const auto& update : jacobian_updates) update();
            }
        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
