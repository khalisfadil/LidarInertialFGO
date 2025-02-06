#include <include/Evaluable/se3/InverseEvaluator.hpp>

namespace slam {
    namespace eval {
        namespace se3 {

            // ----------------------------------------------------------------------------
            // MakeShared
            // ----------------------------------------------------------------------------

            auto InverseEvaluator::MakeShared(const Evaluable<InType>::ConstPtr& transform) -> Ptr {
                return std::make_shared<InverseEvaluator>(transform);
            }

            // ----------------------------------------------------------------------------
            // InverseEvaluator
            // ----------------------------------------------------------------------------

            InverseEvaluator::InverseEvaluator(const Evaluable<InType>::ConstPtr& transform)
                : transform_(transform) {}

            // ----------------------------------------------------------------------------
            // active
            // ----------------------------------------------------------------------------

            bool InverseEvaluator::active() const {
                return transform_->active();
            }

            // ----------------------------------------------------------------------------
            // getRelatedVarKeys
            // ----------------------------------------------------------------------------

            void InverseEvaluator::getRelatedVarKeys(KeySet& keys) const {
                transform_->getRelatedVarKeys(keys);
            }

            // ----------------------------------------------------------------------------
            // value
            // ----------------------------------------------------------------------------

            auto InverseEvaluator::value() const -> OutType {
                return transform_->evaluate().inverse();
            }

            // ----------------------------------------------------------------------------
            // forward
            // ----------------------------------------------------------------------------

            auto InverseEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child = transform_->forward();
                const auto value = child->value().inverse();

                auto node = Node<OutType>::MakeShared(value);
                node->addChild(child);
                return node;
            }

            // ----------------------------------------------------------------------------
            // backward
            // ----------------------------------------------------------------------------

            void InverseEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                    const Node<OutType>::Ptr& node,
                                    StateKeyJacobians& jacs) const {
                if (!node || node->size() < 1) {
                    throw std::runtime_error("[InverseEvaluator::backward] Node has insufficient children.");
                }

                auto child_base = node->getChild(0);
                if (!child_base) {
                    throw std::runtime_error("[InverseEvaluator::backward] Null child node encountered.");
                }

                auto child = std::dynamic_pointer_cast<Node<InType>>(child_base);
                if (!child || !child->hasValue()) {
                    throw std::runtime_error("[InverseEvaluator::backward] Invalid child node.");
                }

                if (transform_->active()) {
                    Eigen::MatrixXd new_lhs = (-1) * lhs * node->value().adjoint();
                    transform_->backward(new_lhs, child, jacs);
                }
            }

            // ----------------------------------------------------------------------------
            // inverse
            // ----------------------------------------------------------------------------

            InverseEvaluator::Ptr inverse(const Evaluable<InverseEvaluator::InType>::ConstPtr& transform) {
                return InverseEvaluator::MakeShared(transform);
            }

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
