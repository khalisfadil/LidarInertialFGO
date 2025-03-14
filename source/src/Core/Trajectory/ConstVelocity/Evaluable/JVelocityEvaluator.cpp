#include "Core/Trajectory/ConstVelocity/Evaluable/JVelocityEvaluator.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto JVelocityEvaluator::MakeShared(
                const eval::Evaluable<XiInType>::ConstPtr& xi,
                const eval::Evaluable<VelInType>::ConstPtr& velocity) -> Ptr {
                return std::make_shared<JVelocityEvaluator>(xi, velocity);
            }

            // -----------------------------------------------------------------------------
            // JVelocityEvaluator
            // -----------------------------------------------------------------------------

            JVelocityEvaluator::JVelocityEvaluator(
                const eval::Evaluable<XiInType>::ConstPtr& xi,
                const eval::Evaluable<VelInType>::ConstPtr& velocity)
                : xi_(xi), velocity_(velocity) {}

                bool JVelocityEvaluator::active() const {
                return xi_->active() || velocity_->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void JVelocityEvaluator::getRelatedVarKeys(eval::Evaluable<XiInType>::KeySet& keys) const {
                xi_->getRelatedVarKeys(keys);
                velocity_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto JVelocityEvaluator::value() const -> OutType {
                return liemath::se3::vec2jac(xi_->evaluate()) * velocity_->evaluate();
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto JVelocityEvaluator::forward() const -> eval::Node<OutType>::Ptr {
                const auto child1 = xi_->forward();
                const auto child2 = velocity_->forward();

                // Compute transformed velocity
                const auto value = liemath::se3::vec2jac(child1->value()) * child2->value();

                // Create a new node and register dependencies
                const auto node = eval::Node<OutType>::MakeShared(value);
                node->addChild(child1);
                node->addChild(child2);

                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void JVelocityEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                            const eval::Node<OutType>::Ptr& node,
                                            eval::StateKeyJacobians& jacs) const {
                // Early exit if neither variable is active
                if (!xi_->active() && !velocity_->active()) return;

                // Retrieve child nodes
                const auto child1 = std::static_pointer_cast<eval::Node<XiInType>>(node->getChild(0));
                const auto child2 = std::static_pointer_cast<eval::Node<VelInType>>(node->getChild(1));

                // Precompute child values
                const auto child1_val = child1->value();
                const auto child2_val = child2->value();

                // Lambda-based Jacobian updates
                std::array<std::function<void()>, 2> updates = {
                    [&]() {
                        if (xi_->active()) {
                            xi_->backward(lhs * (-0.5 * liemath::se3::curlyhat(child2_val)), child1, jacs);
                        }
                    },
                    [&]() {
                        if (velocity_->active()) {
                            velocity_->backward(lhs * liemath::se3::vec2jac(child1_val), child2, jacs);
                        }
                    }
                };

                // Execute updates
                for (const auto& update : updates) update();
            }

            // -----------------------------------------------------------------------------
            // j_velocity
            // -----------------------------------------------------------------------------

            JVelocityEvaluator::Ptr j_velocity(
                const eval::Evaluable<JVelocityEvaluator::XiInType>::ConstPtr& xi,
                const eval::Evaluable<JVelocityEvaluator::VelInType>::ConstPtr& velocity) {
                return JVelocityEvaluator::MakeShared(xi, velocity);
            }
        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
