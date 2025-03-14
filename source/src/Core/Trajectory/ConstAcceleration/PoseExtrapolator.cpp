#include "Core/Trajectory/ConstAcceleration/PoseExtrapolator.hpp"

#include "Core/Trajectory/ConstAcceleration/Helper.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto PoseExtrapolator::MakeShared(const Time& time,
                                              const Variable::ConstPtr& knot1) -> Ptr {
                return std::make_shared<PoseExtrapolator>(time, knot1);
            }

            // -----------------------------------------------------------------------------
            // PoseExtrapolator
            // -----------------------------------------------------------------------------

            PoseExtrapolator::PoseExtrapolator(const Time& time, const Variable::ConstPtr& knot)
            : knot_(knot), Phi_(getTran((time - knot->getTime()).seconds())) {}

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------

            bool PoseExtrapolator::active() const {
                return knot_->getPose()->active() || knot_->getVelocity()->active() || knot_->getAcceleration()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void PoseExtrapolator::getRelatedVarKeys(KeySet& keys) const {
                knot_->getPose()->getRelatedVarKeys(keys);
                knot_->getVelocity()->getRelatedVarKeys(keys);
                knot_->getAcceleration()->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto PoseExtrapolator::value() const -> OutType {
                const auto& vel = knot_->getVelocity()->value();
                const auto& acc = knot_->getAcceleration()->value();
                const slam::liemath::se3::Transformation T_i1(Phi_.block<6, 6>(0, 6) * vel + Phi_.block<6, 6>(0, 12) * acc, 0);
                
                const auto pose_val = knot_->getPose()->value();
                return OutType(T_i1 * pose_val);
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto PoseExtrapolator::forward() const -> eval::Node<OutType>::Ptr {
                const auto T = knot_->getPose()->forward();
                const auto w = knot_->getVelocity()->forward();
                const auto dw = knot_->getAcceleration()->forward();

                // Compute interpolated transformation
                const slam::liemath::se3::Transformation T_i1(Phi_.block<6, 6>(0, 6) * w->value() +
                                                            Phi_.block<6, 6>(0, 12) * dw->value(),0);
                const auto node = eval::Node<OutType>::MakeShared(T_i1 * T->value());

                // Explicitly specify the container type
                std::initializer_list<slam::eval::NodeBase::Ptr> children = {T, w, dw};
                for (const auto& child : children) node->addChild(child);
                
                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void PoseExtrapolator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                              const eval::Node<OutType>::Ptr& node,
                              eval::StateKeyJacobians& jacs) const {
                if (!active()) return;

                // Retrieve state values
                const auto w = knot_->getVelocity()->value();
                const auto dw = knot_->getAcceleration()->value();

                // Compute xi_i1 and transformation
                const Eigen::Matrix<double, 6, 1> xi_i1 = Phi_.block<6, 6>(0, 6) * w + Phi_.block<6, 6>(0, 12) * dw;
                const Eigen::Matrix<double, 6, 6> J_i1 = slam::liemath::se3::vec2jac(xi_i1);
                const slam::liemath::se3::Transformation T_i1(xi_i1,0);

                // Lambda-based Jacobian updates
                std::array<std::function<void()>, 3> jacobian_updates = {
                    [&] { if (knot_->getPose()->active()) 
                            knot_->getPose()->backward(lhs * T_i1.adjoint(), 
                                std::static_pointer_cast<eval::Node<InPoseType>>(node->getChild(0)), jacs); },
                    [&] { if (knot_->getVelocity()->active()) 
                            knot_->getVelocity()->backward(lhs * J_i1 * Phi_.block<6, 6>(0, 6), 
                                std::static_pointer_cast<eval::Node<InVelType>>(node->getChild(1)), jacs); },
                    [&] { if (knot_->getAcceleration()->active()) 
                            knot_->getAcceleration()->backward(lhs * J_i1 * Phi_.block<6, 6>(0, 12), 
                                std::static_pointer_cast<eval::Node<InAccType>>(node->getChild(2)), jacs); }
                };

                for (const auto& update : jacobian_updates) update();
            }

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam