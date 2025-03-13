#pragma once

#include <Eigen/Core>

#include "LGMath/LieGroupMath.hpp"
#include "Evaluable/Evaluable.hpp"

namespace slam {
    namespace eval {
        namespace se3 {
            
            // -----------------------------------------------------------------------------
            /**
             * @class ComposeInverseEval
             * @brief Computes the composition of two SE(3) transformations where the second is inverted.
             *
             * Given two transformations \( T_1 \) and \( T_2 \), this evaluator computes:
             * \f[
             * T_{result} = T_1 \cdot T_2^{-1}
             * \f]
             * It supports automatic differentiation, allowing it to be used in optimization problems.
             */
            class ComposeInverseEvaluator : public Evaluable<slam::liemath::se3::Transformation> {
            public:
            using Ptr = std::shared_ptr<ComposeInverseEvaluator>;
            using ConstPtr = std::shared_ptr<const ComposeInverseEvaluator>;

            using InType = slam::liemath::se3::Transformation;
            using OutType = slam::liemath::se3::Transformation;

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory method to create a shared pointer to ComposeInverseEval.
             * @param transform1 First transformation \( T_1 \).
             * @param transform2 Second transformation \( T_2 \).
             * @return Shared pointer to a new ComposeInverseEval instance.
             */
            static Ptr MakeShared(const Evaluable<InType>::ConstPtr& transform1,
                                    const Evaluable<InType>::ConstPtr& transform2);

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs a ComposeInverseEval evaluator.
             * @param transform1 First transformation \( T_1 \).
             * @param transform2 Second transformation \( T_2 \).
             */
            ComposeInverseEvaluator(const Evaluable<InType>::ConstPtr& transform1,
                                const Evaluable<InType>::ConstPtr& transform2);

            // -----------------------------------------------------------------------------
            /**
             * @brief Checks if either of the transformations depends on active variables.
             * @return True if at least one transformation is active, otherwise false.
             */
            bool active() const override;

            // -----------------------------------------------------------------------------
            /**
             * @brief Collects the keys of all state variables influencing this evaluator.
             * @param[out] keys The set of state keys related to this evaluator.
             */
            void getRelatedVarKeys(KeySet &keys) const override;
            
            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the composed transformation \( T_1 \cdot T_2^{-1} \).
             * @return The resulting SE(3) transformation.
             */
            OutType value() const override;

            // -----------------------------------------------------------------------------
            /**
             * @brief Forward pass: Computes and stores the transformation for automatic differentiation.
             * @return A node containing the computed transformation \( T_1 \cdot T_2^{-1} \).
             */
            Node<OutType>::Ptr forward() const override;

            // -----------------------------------------------------------------------------
            /**
             * @brief Backward pass: Computes and accumulates Jacobians for optimization.
             * 
             * Given a left-hand side (LHS) weight matrix and a node from the forward pass,
             * this method propagates gradients to both input transformations.
             * 
             * @param lhs Left-hand-side weight matrix.
             * @param node Node containing the forward-pass result.
             * @param jacs Container to store the computed Jacobians.
             */
            void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                            const Node<OutType>::Ptr& node,
                            StateKeyJacobians& jacs) const override;

            private:
            const Evaluable<InType>::ConstPtr transform1_; ///< First transformation \( T_1 \).
            const Evaluable<InType>::ConstPtr transform2_; ///< Second transformation \( T_2 \).
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Creates a ComposeInverseEval evaluator.
             *
             * This is a convenience function to create an evaluator for computing \( T_1 \cdot T_2^{-1} \).
             * 
             * @param transform1 First transformation \( T_1 \).
             * @param transform2 Second transformation \( T_2 \).
             * @return Shared pointer to the created evaluator.
             */
            ComposeInverseEvaluator::Ptr compose_rinv(
                const Evaluable<ComposeInverseEvaluator::InType>::ConstPtr& transform1,
                const Evaluable<ComposeInverseEvaluator::InType>::ConstPtr& transform2);

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
