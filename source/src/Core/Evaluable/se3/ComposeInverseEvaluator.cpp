#include "Core/Evaluable/se3/ComposeInverseEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace se3 {
            
            // -----------------------------------------------------------------------------
            // Factory method to create a shared pointer instance.
            // -----------------------------------------------------------------------------

            auto ComposeInverseEvaluator::MakeShared(
                const Evaluable<InType>::ConstPtr& transform1,
                const Evaluable<InType>::ConstPtr& transform2) -> Ptr {
                return std::make_shared<ComposeInverseEvaluator>(transform1, transform2);
            }

            // -----------------------------------------------------------------------------
            // Constructor that initializes the transformations.
            // -----------------------------------------------------------------------------

            ComposeInverseEvaluator::ComposeInverseEvaluator(
                const Evaluable<InType>::ConstPtr& transform1,
                const Evaluable<InType>::ConstPtr& transform2)
                : transform1_(transform1), transform2_(transform2) {}

            // -----------------------------------------------------------------------------
            // Checks if either transformation is active.
            // -----------------------------------------------------------------------------

            bool ComposeInverseEvaluator::active() const {
                return transform1_->active() || transform2_->active();
            }

            // -----------------------------------------------------------------------------
            // Retrieves the set of keys associated with the transformations.
            // -----------------------------------------------------------------------------

            void ComposeInverseEvaluator::getRelatedVarKeys(KeySet &keys) const {
                transform1_->getRelatedVarKeys(keys);
                transform2_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // Computes the composed transformation T1 * T2⁻¹.
            // -----------------------------------------------------------------------------

            auto ComposeInverseEvaluator::value() const -> OutType {
                return transform1_->evaluate() * transform2_->evaluate().inverse();
            }

            // -----------------------------------------------------------------------------
            // Forward pass for computing the transformation.
            // -----------------------------------------------------------------------------

            auto ComposeInverseEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child1 = transform1_->forward();
                const auto child2 = transform2_->forward();
                const auto value = child1->value() * child2->value().inverse();
                
                // Store result in node
                auto node = Node<OutType>::MakeShared(value);
                node->addChild(child1);
                node->addChild(child2);
                return node;
            }

            // -----------------------------------------------------------------------------
            // Backward pass for Jacobian computation.
            // -----------------------------------------------------------------------------

            void ComposeInverseEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                                const Node<OutType>::Ptr& node,
                                                StateKeyJacobians& jacs) const {
                // Ensure the node exists and has at least two children
                if (!node || node->size() < 2) {
                    throw std::runtime_error("[ComposeInverseEvaluator::backward] Node has insufficient children.");
                }

                // Retrieve child nodes safely using `getChild`
                auto child1_base = node->getChild(0);
                auto child2_base = node->getChild(1);

                // Ensure the retrieved child nodes are valid
                if (!child1_base || !child2_base) {
                    throw std::runtime_error("[ComposeInverseEvaluator::backward] Null child node encountered.");
                }

                // Attempt to cast to `Node<InType>` (ensure proper type conversion)
                auto child1 = std::static_pointer_cast<Node<InType>>(child1_base);
                auto child2 = std::static_pointer_cast<Node<InType>>(child2_base);

                // Check if casting was successful
                if (!child1 || !child2) {
                    throw std::runtime_error("[ComposeInverseEvaluator::backward] Child node type mismatch.");
                }

                // Ensure the child nodes contain valid transformation values
                if (!child1->hasValue() || !child2->hasValue()) {
                    throw std::runtime_error("[ComposeInverseEvaluator::backward] Child node has no value.");
                }

                // Compute Jacobian for transform1
                if (transform1_->active()) {
                    transform1_->backward(lhs, child1, jacs);
                }

                // Compute Jacobian for transform2 using SE(3) Adjoint
                if (transform2_->active()) {
                    // Compute transformation \( T_{ba} = T_1 \cdot T_2^{-1} \)
                    const auto T_ba = child1->value() * child2->value().inverse();

                    // Compute adjoint transformation for efficient Jacobian propagation
                    Eigen::MatrixXd adj_T_ba_inv = T_ba.adjoint(); 
                    Eigen::MatrixXd new_lhs = -lhs * adj_T_ba_inv;

                    // Compute backward Jacobian for transform2
                    transform2_->backward(new_lhs, child2, jacs);
                }
            }

            // -----------------------------------------------------------------------------
            // Convenience function to create a ComposeInverseEval instance.
            // -----------------------------------------------------------------------------

            ComposeInverseEvaluator::Ptr compose_rinv(const Evaluable<ComposeInverseEvaluator::InType>::ConstPtr& transform1,
                                                const Evaluable<ComposeInverseEvaluator::InType>::ConstPtr& transform2) {
                return ComposeInverseEvaluator::MakeShared(transform1, transform2);
            }

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
