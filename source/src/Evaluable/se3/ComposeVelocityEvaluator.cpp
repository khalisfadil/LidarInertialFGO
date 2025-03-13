#include "Evaluable/se3/ComposeVelocityEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace se3 {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto ComposeVelocityEvaluator::MakeShared(
                const Evaluable<PoseInType>::ConstPtr& transform,
                const Evaluable<VelInType>::ConstPtr& velocity) -> Ptr {
                return std::make_shared<ComposeVelocityEvaluator>(transform, velocity);
            }

            // -----------------------------------------------------------------------------
            // ComposeVelocityEvaluator
            // -----------------------------------------------------------------------------

            ComposeVelocityEvaluator::ComposeVelocityEvaluator(
                const Evaluable<PoseInType>::ConstPtr& transform,
                const Evaluable<VelInType>::ConstPtr& velocity)
                : transform_(transform), velocity_(velocity) {}

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------

            bool ComposeVelocityEvaluator::active() const {
                return transform_->active() || velocity_->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void ComposeVelocityEvaluator::getRelatedVarKeys(KeySet& keys) const {
                transform_->getRelatedVarKeys(keys);
                velocity_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto ComposeVelocityEvaluator::value() const -> OutType {
                return slam::liemath::se3::tranAd(transform_->evaluate().matrix()) * velocity_->evaluate();
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto ComposeVelocityEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child1 = transform_->forward();
                const auto child2 = velocity_->forward();
                const auto value = slam::liemath::se3::tranAd(child1->value().matrix()) * child2->value();

                auto node = Node<OutType>::MakeShared(value);
                node->addChild(child1);
                node->addChild(child2);
                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void ComposeVelocityEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                            const Node<OutType>::Ptr& node,
                                            StateKeyJacobians& jacs) const {
                if (!node || node->size() < 2) {
                    throw std::runtime_error("[ComposeVelocityEvaluator::backward] Node has insufficient children.");
                }

                auto child1_base = node->getChild(0);
                auto child2_base = node->getChild(1);

                if (!child1_base || !child2_base) {
                    throw std::runtime_error("[ComposeVelocityEvaluator::backward] Null child node encountered.");
                }

                auto child1 = std::static_pointer_cast<Node<PoseInType>>(child1_base);
                auto child2 = std::static_pointer_cast<Node<VelInType>>(child2_base);

                if (!child1 || !child2 || !child1->hasValue() || !child2->hasValue()) {
                    throw std::runtime_error("[ComposeVelocityEvaluator::backward] Invalid child node.");
                }

                if (transform_->active()) {
                    transform_->backward(-lhs * slam::liemath::se3::curlyhat(node->value()), child1, jacs);
                }

                if (velocity_->active()) {
                    velocity_->backward(lhs * slam::liemath::se3::tranAd(child1->value().matrix()), child2, jacs);
                }
            }

            // -----------------------------------------------------------------------------
            // compose_velocity
            // -----------------------------------------------------------------------------

            ComposeVelocityEvaluator::Ptr compose_velocity(
                const Evaluable<ComposeVelocityEvaluator::PoseInType>::ConstPtr& transform,
                const Evaluable<ComposeVelocityEvaluator::VelInType>::ConstPtr& velocity) {
                return ComposeVelocityEvaluator::MakeShared(transform, velocity);
            }

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
