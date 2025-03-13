#pragma once

#include <Eigen/Core>
#include <memory>

#include "Evaluable/Evaluable.hpp"
#include "LGMath/LieGroupMath.hpp"

namespace slam {
    namespace eval {
        namespace se3 {

            // -----------------------------------------------------------------------------
            /**
             * @class ComposeEvaluator
             * @brief Computes the composition of two SE(3) transformations in a factor graph.
             *
             * This class represents an evaluable function that composes two SE(3) transformations
             * using Lie group operations. It is commonly used in **pose graph optimization**,
             * **SLAM**, and **robot state estimation**, where transformations need to be chained
             * efficiently while allowing automatic differentiation.
             *
             * Given two transformations:
             *  - **T1**: First SE(3) transformation
             *  - **T2**: Second SE(3) transformation
             *
             * The composition is computed as:
             *   \f$ T = T1 \cdot T2 \f$
             *
             * This evaluator ensures that gradients (Jacobians) are properly propagated for
             * optimization-based estimation.
             */
            class ComposeEvaluator : public Evaluable<slam::liemath::se3::Transformation> {

                public:

                    using Ptr = std::shared_ptr<ComposeEvaluator>;
                    using ConstPtr = std::shared_ptr<const ComposeEvaluator>;

                    using InType = slam::liemath::se3::Transformation;  ///< Input type: SE(3) transformation
                    using OutType = slam::liemath::se3::Transformation; ///< Output type: SE(3) transformation

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared instance of ComposeEvaluator.
                     * 
                     * This ensures efficient memory management when creating a ComposeEvaluator instance.
                     *
                     * @param transform1 First SE(3) transformation.
                     * @param transform2 Second SE(3) transformation.
                     * @return Shared pointer to the newly created ComposeEvaluator.
                     */
                    static Ptr MakeShared(const Evaluable<InType>::ConstPtr& transform1,
                                            const Evaluable<InType>::ConstPtr& transform2);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor for ComposeEvaluator.
                     * 
                     * Initializes the evaluator with two input transformations.
                     *
                     * @param transform1 First SE(3) transformation.
                     * @param transform2 Second SE(3) transformation.
                     */
                    ComposeEvaluator(const Evaluable<InType>::ConstPtr& transform1,
                                    const Evaluable<InType>::ConstPtr& transform2);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Checks if the evaluator depends on active variables.
                     * 
                     * This function determines whether either of the input transformations
                     * is involved in optimization. If so, this evaluator should be considered
                     * active in the factor graph.
                     *
                     * @return True if at least one transformation is active, otherwise false.
                     */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Gathers state variables that influence this evaluator.
                     * 
                     * This function retrieves the state keys associated with `transform1_` and `transform2_`,
                     * allowing factor graph solvers to identify dependencies.
                     *
                     * @param keys Set of state keys to be populated.
                     */
                    void getRelatedVarKeys(KeySet& keys) const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the composed SE(3) transformation.
                     * 
                     * This function retrieves the values of `transform1_` and `transform2_`, then performs
                     * SE(3) composition (`T1 * T2`), returning the result.
                     *
                     * @return Composed SE(3) transformation.
                     */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Forward pass for computational graph construction.
                     * 
                     * Creates a new computational node representing the composed transformation and
                     * links it to its parent nodes (`transform1_` and `transform2_`). This is used
                     * for automatic differentiation in optimization.
                     *
                     * @return A node in the computational graph holding the transformation result.
                     */
                    Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Backward pass for gradient computation.
                     * 
                     * Computes the Jacobians of the transformation composition and accumulates them
                     * for optimization. The gradients are computed w.r.t both input transformations.
                     *
                     * @param lhs Left-hand-side weight matrix for chain rule computation.
                     * @param node Computational graph node corresponding to this evaluator.
                     * @param jacs Container to store accumulated Jacobians.
                     */
                    void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                    const Node<OutType>::Ptr& node,
                                    StateKeyJacobians& jacs) const override;

                private:

                    const Evaluable<InType>::ConstPtr transform1_; ///< First transformation (SE(3))
                    const Evaluable<InType>::ConstPtr transform2_; ///< Second transformation (SE(3))
            };

            /**
             * @brief Convenience function to compose two SE(3) transformations.
             * 
             * Instead of manually instantiating `ComposeEvaluator`, this helper function provides
             * a clean and intuitive way to create a composition evaluator.
             *
             * @param transform1 First SE(3) transformation.
             * @param transform2 Second SE(3) transformation.
             * @return Shared pointer to a ComposeEvaluator instance.
             */
            ComposeEvaluator::Ptr compose(
                const Evaluable<ComposeEvaluator::InType>::ConstPtr& transform1,
                const Evaluable<ComposeEvaluator::InType>::ConstPtr& transform2);

        }  // namespace se3
    }   // namespace eval
}  // namespace slam
