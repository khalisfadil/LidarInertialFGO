#include "source/include/Evaluable/se3/LogMapEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace se3 {
            
            // ----------------------------------------------------------------------------
            // MakeShared
            // ----------------------------------------------------------------------------

            auto LogMapEvaluator::MakeShared(const Evaluable<InType>::ConstPtr& transform) -> Ptr {
                return std::make_shared<LogMapEvaluator>(transform);
            }

            // ----------------------------------------------------------------------------
            // LogMapEvaluator
            // ----------------------------------------------------------------------------

            LogMapEvaluator::LogMapEvaluator(const Evaluable<InType>::ConstPtr& transform)
                : transform_(transform) {}

            // ----------------------------------------------------------------------------
            // active
            // ----------------------------------------------------------------------------

            bool LogMapEvaluator::active() const {
                return transform_->active();
            }

            // ----------------------------------------------------------------------------
            // getRelatedVarKeys
            // ----------------------------------------------------------------------------

            void LogMapEvaluator::getRelatedVarKeys(KeySet& keys) const {
                transform_->getRelatedVarKeys(keys);
            }

            // ----------------------------------------------------------------------------
            // value
            // ----------------------------------------------------------------------------

            auto LogMapEvaluator::value() const -> OutType {
                return transform_->evaluate().vec();
            }

            // ----------------------------------------------------------------------------
            // forward
            // ----------------------------------------------------------------------------

            auto LogMapEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child = transform_->forward();
                const auto value = child->value().vec();

                auto node = Node<OutType>::MakeShared(value);
                node->addChild(child);
                return node;
            }

            // ----------------------------------------------------------------------------
            // backward
            // ----------------------------------------------------------------------------

            void LogMapEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                    const Node<OutType>::Ptr& node,
                                    StateKeyJacobians& jacs) const {
                if (!node || node->size() < 1) {
                    throw std::runtime_error("[LogMapEvaluator::backward] Node has insufficient children.");
                }

                auto child_base = node->getChild(0);
                if (!child_base) {
                    throw std::runtime_error("[LogMapEvaluator::backward] Null child node encountered.");
                }

                auto child = std::static_pointer_cast<Node<InType>>(child_base);
                if (!child || !child->hasValue()) {
                    throw std::runtime_error("[LogMapEvaluator::backward] Invalid child node.");
                }

                if (transform_->active()) {
                    Eigen::MatrixXd new_lhs = lhs * slam::liemath::se3::vec2jacinv(node->value());
                    transform_->backward(new_lhs, child, jacs);
                }
            }

            // ----------------------------------------------------------------------------
            // tran2vec
            // ----------------------------------------------------------------------------

            LogMapEvaluator::Ptr tran2vec(const Evaluable<LogMapEvaluator::InType>::ConstPtr& transform) {
                return LogMapEvaluator::MakeShared(transform);
            }
        }  // namespace se3
    }  // namespace eval
}  // namespace slam
