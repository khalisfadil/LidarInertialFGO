#pragma once

#include <Eigen/Core>

#include "LGMath/LieGroupMath.hpp"
#include "Core/Evaluable/Evaluable.hpp"

namespace slam {
    namespace eval {
        namespace se3 {

            // -----------------------------------------------------------------------------
            /**
             * @class ExpMapEval
             * @brief Computes the exponential map from a twist vector \( \xi \) to an SE(3) transformation.
             *
             * Given a 6D twist vector \( \xi \) (Lie algebra element), this evaluator computes:
             * \f[
             * T = \exp(\hat{\xi})
             * \f]
             * where \( \exp(\hat{\xi}) \) maps from \( \mathfrak{se}(3) \) (Lie algebra) to \( SE(3) \) (Lie group).
             * This is crucial for motion models, pose integration, and SLAM optimization.
             */
            class ExpMapEvaluator : public Evaluable<slam::liemath::se3::Transformation> {
                public:
                    using Ptr = std::shared_ptr<ExpMapEvaluator>;
                    using ConstPtr = std::shared_ptr<const ExpMapEvaluator>;

                    using InType = Eigen::Matrix<double, 6, 1>;
                    using OutType = slam::liemath::se3::Transformation;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared pointer instance.
                     * @param xi The twist vector \( \xi \).
                     * @return Shared pointer to a new ExpMapEvaluator instance.
                     */
                    static Ptr MakeShared(const Evaluable<InType>::ConstPtr& xi);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor that initializes the evaluator.
                     * @param xi The twist vector \( \xi \).
                     */
                    ExpMapEvaluator(const Evaluable<InType>::ConstPtr& xi);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Checks if the twist vector \( \xi \) depends on active state variables.
                     * @return True if \( \xi \) is active, otherwise false.
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
                     * @brief Computes the SE(3) transformation \( T = \exp(\hat{\xi}) \).
                     * @return The resulting SE(3) transformation.
                     */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Forward pass: Computes and stores the transformation.
                     * @return A node containing the computed SE(3) transformation.
                     */
                    Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Backward pass: Computes and accumulates Jacobians for optimization.
                     * 
                     * Given a left-hand side (LHS) weight matrix and a node from the forward pass,
                     * this method propagates gradients to the twist vector \( \xi \).
                     * 
                     * @param lhs Left-hand-side weight matrix.
                     * @param node Node containing the forward-pass result.
                     * @param jacs Container to store the computed Jacobians.
                     */
                    void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                const Node<OutType>::Ptr& node,
                                StateKeyJacobians& jacs) const override;

                private:
                    const Evaluable<InType>::ConstPtr xi_; ///< Twist vector \( \xi \).
                };

                // -----------------------------------------------------------------------------
                /**
                 * @brief Creates an ExpMapEvaluator evaluator.
                 *
                 * This is a convenience function to create an evaluator for computing \( T = \exp(\hat{\xi}) \).
                 * 
                 * @param xi The twist vector \( \xi \).
                 * @return Shared pointer to the created evaluator.
                 */
                ExpMapEvaluator::Ptr vec2tran(const Evaluable<ExpMapEvaluator::InType>::ConstPtr& xi);

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
