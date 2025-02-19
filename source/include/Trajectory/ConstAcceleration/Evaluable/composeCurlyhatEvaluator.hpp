#pragma once

#include <Eigen/Core>
#include <memory>

#include "source/include/LGMath/LieGroupMath.hpp"

#include "source/include/Evaluable/Evaluable.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            /**
             * @class ComposeCurlyhatEvaluator
             * @brief Evaluates the product of the `curlyhat` operation applied to two SE(3) vectors.
             *
             * Given two input vectors `x` and `y` in `se(3)`, this evaluator computes:
             * \f[
             * \hat{x} \cdot y
             * \f]
             * where `\hat{x}` represents the `curlyhat` operator that maps `se(3)` vectors
             * to their corresponding Lie algebra matrix representation.
             *
             * This evaluator supports **automatic differentiation**, **forward evaluation**, 
             * and **Jacobian computation** for factor graph optimization.
             */
            class ComposeCurlyhatEvaluator : public slam::eval::Evaluable<Eigen::Matrix<double, 6, 1>> {
                public:
                    using Ptr = std::shared_ptr<ComposeCurlyhatEvaluator>;
                    using ConstPtr = std::shared_ptr<const ComposeCurlyhatEvaluator>;

                    using InType = Eigen::Matrix<double, 6, 1>;
                    using OutType = Eigen::Matrix<double, 6, 1>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared instance of `ComposeCurlyhatEvaluator`.
                     * @param x First input evaluator (se(3) vector).
                     * @param y Second input evaluator (se(3) vector).
                     * @return Shared pointer to a new `ComposeCurlyhatEvaluator` instance.
                     */
                    static Ptr MakeShared(const Evaluable<InType>::ConstPtr& x,
                                        const Evaluable<InType>::ConstPtr& y);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructs a `ComposeCurlyhatEvaluator` with two `se(3)` input evaluators.
                     * @param x First input evaluator.
                     * @param y Second input evaluator.
                     */
                    ComposeCurlyhatEvaluator(const Evaluable<InType>::ConstPtr& x,
                                            const Evaluable<InType>::ConstPtr& y);

                    // -----------------------------------------------------------------------------
                    /** @brief Checks if the evaluator depends on any active variables. */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Retrieves related variable keys for factor graph optimization. */
                    void getRelatedVarKeys(KeySet& keys) const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Computes the evaluated output. */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Computes the forward evaluation and returns the result as a node. */
                    eval::Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /** 
                     * @brief Computes the backward pass, accumulating Jacobians.
                     * @param lhs Left-hand side of the Jacobian computation.
                     * @param node Node representing this evaluator.
                     * @param jacs Container for accumulated Jacobians.
                     */
                    void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs, 
                                const slam::eval::Node<OutType>::Ptr& node,
                                slam::eval::StateKeyJacobians& jacs) const override;

                private:
                        const Evaluable<InType>::ConstPtr x_;  ///< First input evaluator (se(3) vector).
                        const Evaluable<InType>::ConstPtr y_;  ///< Second input evaluator (se(3) vector).
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory function to create a `ComposeCurlyhatEvaluator` instance.
             * @param x First input evaluator.
             * @param y Second input evaluator.
             * @return Shared pointer to a `ComposeCurlyhatEvaluator`.
             */
            ComposeCurlyhatEvaluator::Ptr compose_curlyhat(
                const slam::eval::Evaluable<ComposeCurlyhatEvaluator::InType>::ConstPtr& x,
                const slam::eval::Evaluable<ComposeCurlyhatEvaluator::InType>::ConstPtr& y);

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
