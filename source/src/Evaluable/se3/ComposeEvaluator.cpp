#include "Evaluable/se3/ComposeEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace se3 {
            
            // -----------------------------------------------------------------------------
            // Factory method to create a shared instance of ComposeEvaluator.
            // -----------------------------------------------------------------------------

            ComposeEvaluator::Ptr ComposeEvaluator::MakeShared(
                const Evaluable<InType>::ConstPtr& transform1,
                const Evaluable<InType>::ConstPtr& transform2) {
                return std::make_shared<ComposeEvaluator>(transform1, transform2);
            }

            // -----------------------------------------------------------------------------
            // Constructor that initializes the two transformations.
            // -----------------------------------------------------------------------------

            ComposeEvaluator::ComposeEvaluator(
                const Evaluable<InType>::ConstPtr& transform1,
                const Evaluable<InType>::ConstPtr& transform2)
                : transform1_(transform1), transform2_(transform2) {}

            // -----------------------------------------------------------------------------
            // Checks if this evaluator depends on active state variables.
            // -----------------------------------------------------------------------------

            bool ComposeEvaluator::active() const {
                return transform1_->active() || transform2_->active();
            }

            // -----------------------------------------------------------------------------
            // Gathers state variable keys that influence this evaluator.
            // -----------------------------------------------------------------------------
            
            void ComposeEvaluator::getRelatedVarKeys(KeySet& keys) const {
                transform1_->getRelatedVarKeys(keys);
                transform2_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // Computes the composed SE(3) transformation.
            // -----------------------------------------------------------------------------

            auto ComposeEvaluator::value() const -> OutType {
                return transform1_->evaluate() * transform2_->evaluate();
            }

            // -----------------------------------------------------------------------------
            // Forward pass for computational graph construction.
            // -----------------------------------------------------------------------------

            auto ComposeEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child1 = transform1_->forward();
                const auto child2 = transform2_->forward();

                if (!child1 || !child2) {
                    throw std::runtime_error("[ComposeEvaluator::forward] Null child node encountered.");
                }

                const auto value = child1->value() * child2->value();
                auto node = Node<OutType>::MakeShared(value);
                node->addChild(child1);
                node->addChild(child2);

                return node;
            }

            // -----------------------------------------------------------------------------
            // Backward pass for Jacobian computation.
            // -----------------------------------------------------------------------------

            void ComposeEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                const Node<OutType>::Ptr& node,
                                StateKeyJacobians& jacs) const {
                // Ensure the node exists and has at least two children
                if (!node || node->size() < 2) {
                    throw std::runtime_error("[ComposeEvaluator::backward] Node has insufficient children.");
                }

                // Retrieve child nodes safely using `getChild`
                auto child1_base = node->getChild(0);
                auto child2_base = node->getChild(1);

                // Ensure the retrieved child nodes are valid
                if (!child1_base || !child2_base) {
                    throw std::runtime_error("[ComposeEvaluator::backward] Null child node encountered.");
                }

                // Attempt to cast to `Node<OutType>` (ensure Node<T> inherits from NodeBase)
                auto child1 = std::static_pointer_cast<Node<OutType>>(child1_base);
                auto child2 = std::static_pointer_cast<Node<OutType>>(child2_base);

                // Check if casting was successful
                if (!child1 || !child2) {
                    throw std::runtime_error("[ComposeEvaluator::backward] Child node type mismatch.");
                }

                // Ensure the child nodes contain valid transformation values
                if (!child1->hasValue() || !child2->hasValue()) {
                    throw std::runtime_error("[ComposeEvaluator::backward] Child node has no value.");
                }

                // Compute Jacobian for transform1
                if (transform1_->active()) {
                    transform1_->backward(lhs, child1, jacs);
                }

                // Compute Jacobian for transform2 using SE(3) Adjoint
                if (transform2_->active()) {
                    Eigen::MatrixXd adj_T1_inv = child1->value().adjoint();  // SE(3) Adjoint
                    Eigen::MatrixXd new_lhs = lhs * adj_T1_inv;

                    transform2_->backward(new_lhs, child2, jacs);
                }
            }

            // -----------------------------------------------------------------------------
            // Convenience function to create a ComposeEvaluator instance.
            // -----------------------------------------------------------------------------

            ComposeEvaluator::Ptr compose(
                const Evaluable<ComposeEvaluator::InType>::ConstPtr& transform1,
                const Evaluable<ComposeEvaluator::InType>::ConstPtr& transform2) {
                return ComposeEvaluator::MakeShared(transform1, transform2);
            }

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
