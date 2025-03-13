#pragma once

#include <Eigen/Core>

#include "LGMath/LieGroupMath.hpp"
#include "Evaluable/Evaluable.hpp"

namespace slam {
    namespace eval {
        namespace se3 {

            // -----------------------------------------------------------------------------
            /**
             * @class InverseEvaluator
             * @brief Computes the inverse of an SE(3) transformation.
             *
             * Given a transformation \( T \), this evaluator computes:
             * \f[
             * T^{-1}
             * \f]
             * where \( T^{-1} \) is the inverse of the rigid body transformation.
             * This is crucial for computing relative poses in SLAM and factor graph optimization.
             */
            class InverseEvaluator : public Evaluable<slam::liemath::se3::Transformation> {
                public:
                    using Ptr = std::shared_ptr<InverseEvaluator>;
                    using ConstPtr = std::shared_ptr<const InverseEvaluator>;

                    using InType = slam::liemath::se3::Transformation;
                    using OutType = slam::liemath::se3::Transformation;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared pointer instance.
                     * @param transform The SE(3) transformation \( T \).
                     * @return Shared pointer to a new InverseEvaluator instance.
                     */
                    static Ptr MakeShared(const Evaluable<InType>::ConstPtr& transform);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor that initializes the evaluator.
                     * @param transform The SE(3) transformation \( T \).
                     */
                    InverseEvaluator(const Evaluable<InType>::ConstPtr& transform);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Checks if the transformation \( T \) depends on active state variables.
                     * @return True if \( T \) is active, otherwise false.
                     */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Collects the state variable keys influencing this evaluator.
                     * @param[out] keys The set of state keys related to this evaluator.
                     */
                    void getRelatedVarKeys(KeySet &keys) const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the inverse transformation \( T^{-1} \).
                     * @return The resulting inverse SE(3) transformation.
                     */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Forward pass: Computes and stores the inverse transformation.
                     * @return A node containing the computed inverse SE(3) transformation.
                     */
                    Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Backward pass: Computes and accumulates Jacobians for optimization.
                     * 
                     * Given a left-hand side (LHS) weight matrix and a node from the forward pass,
                     * this method propagates gradients to the transformation \( T \).
                     * 
                     * @param lhs Left-hand-side weight matrix.
                     * @param node Node containing the forward-pass result.
                     * @param jacs Container to store the computed Jacobians.
                     */
                    void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                const Node<OutType>::Ptr& node,
                                StateKeyJacobians& jacs) const override;

                private:
                    const Evaluable<InType>::ConstPtr transform_; ///< SE(3) transformation \( T \).
                };

                // -----------------------------------------------------------------------------
                /**
                 * @brief Creates an InverseEvaluator evaluator.
                 *
                 * This is a convenience function to create an evaluator for computing \( T^{-1} \).
                 * 
                 * @param transform The SE(3) transformation \( T \).
                 * @return Shared pointer to the created evaluator.
                 */
                InverseEvaluator::Ptr inverse(const Evaluable<InverseEvaluator::InType>::ConstPtr& transform);

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
