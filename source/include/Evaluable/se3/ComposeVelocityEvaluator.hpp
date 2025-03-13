#pragma once

#include <Eigen/Core>

#include "LGMath/LieGroupMath.hpp"
#include "Evaluable/Evaluable.hpp"

namespace slam {
    namespace eval {
        namespace se3 {

            // -----------------------------------------------------------------------------
            /**
             * @class ComposeVelocityEval
             * @brief Computes the transformed velocity under an SE(3) transformation.
             *
             * Given a pose transformation \( T \) and a velocity \( \xi \), this evaluator computes:
             * \f[
             * \xi' = \text{tranAd}(T) \cdot \xi
             * \f]
             * This operation transforms a local frame velocity \( \xi \) to the world frame.
             * It supports automatic differentiation for optimization applications.
             */
            class ComposeVelocityEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
                public:
                    using Ptr = std::shared_ptr<ComposeVelocityEvaluator>;
                    using ConstPtr = std::shared_ptr<const ComposeVelocityEvaluator>;

                    using PoseInType = slam::liemath::se3::Transformation;
                    using VelInType = Eigen::Matrix<double, 6, 1>;
                    using OutType = Eigen::Matrix<double, 6, 1>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared pointer instance.
                     * @param transform The SE(3) transformation \( T \).
                     * @param velocity The velocity vector \( \xi \).
                     * @return Shared pointer to a new ComposeVelocityEval instance.
                     */
                    static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr& transform,
                                        const Evaluable<VelInType>::ConstPtr& velocity);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor that initializes the transformations.
                     * @param transform The SE(3) transformation \( T \).
                     * @param velocity The velocity vector \( \xi \).
                     */
                    ComposeVelocityEvaluator(const Evaluable<PoseInType>::ConstPtr& transform,
                                        const Evaluable<VelInType>::ConstPtr& velocity);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Checks if either of the inputs depends on active state variables.
                     * @return True if at least one input is active, otherwise false.
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
                     * @brief Computes the transformed velocity \( \xi' \).
                     * @return The resulting transformed velocity.
                     */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Forward pass: Computes and stores the transformed velocity.
                     * @return A node containing the computed transformed velocity.
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
                    const Evaluable<PoseInType>::ConstPtr transform_; ///< SE(3) transformation \( T \).
                    const Evaluable<VelInType>::ConstPtr velocity_; ///< 6D velocity \( \xi \).
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Creates a ComposeVelocityEval evaluator.
             *
             * This is a convenience function to create an evaluator for computing \( \xi' = \text{tranAd}(T) \cdot \xi \).
             * 
             * @param transform The SE(3) transformation \( T \).
             * @param velocity The velocity vector \( \xi \).
             * @return Shared pointer to the created evaluator.
             */
            ComposeVelocityEvaluator::Ptr compose_velocity(
                const Evaluable<ComposeVelocityEvaluator::PoseInType>::ConstPtr& transform,
                const Evaluable<ComposeVelocityEvaluator::VelInType>::ConstPtr& velocity);

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
