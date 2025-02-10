#pragma once

#include <memory>
#include <Eigen/Core>

#include "source/include/Evaluable/Evaluable.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // -----------------------------------------------------------------------------
            /**
             * @class ScalarMultEvaluator
             * @brief Evaluates the element-wise scalar multiplication of a vector function.
             *
             * **Functionality:**
             * - Computes **s * v**, where `s` is a scalar and `v` is a vector-valued function.
             * - Supports forward and backward propagation for **automatic differentiation**.
             *
             * @tparam DIM Dimensionality of the input vector (can be dynamic).
             */
            template <int DIM = Eigen::Dynamic>
            class ScalarMultEvaluator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
                public:
                    using Ptr = std::shared_ptr<ScalarMultEvaluator>;
                    using ConstPtr = std::shared_ptr<const ScalarMultEvaluator>;

                    using InType = Eigen::Matrix<double, DIM, 1>;
                    using OutType = Eigen::Matrix<double, DIM, 1>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method for creating a shared instance.
                     * @param v Input vector function.
                     * @param s Scalar multiplier.
                     */
                    static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v, const double& s);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor for ScalarMultEvaluator.
                     * @param v Input vector function.
                     * @param s Scalar multiplier.
                     */
                    ScalarMultEvaluator(const typename Evaluable<InType>::ConstPtr& v, const double& s);

                    // -----------------------------------------------------------------------------
                    /** @brief Checks if this evaluator depends on active state variables. */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Retrieves keys of related state variables. */
                    void getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Computes the scalar-multiplied output value. */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Creates a computation node for forward propagation. */
                    typename Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Performs backpropagation to accumulate Jacobians.
                     * @param lhs Left-hand-side weight matrix.
                     * @param node Computation node from `forward()`.
                     * @param jacs Storage for accumulated Jacobians.
                     */
                    void backward(const Eigen::MatrixXd& lhs,
                                  const typename Node<OutType>::Ptr& node,
                                  StateKeyJacobians& jacs) const override;

                private:
                    const typename Evaluable<InType>::ConstPtr v_;  ///< Input vector function.
                    const double s_;  ///< Scalar multiplier.
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Convenience function for creating a ScalarMultEvaluator instance.
             * @param v Input vector function.
             * @param s Scalar multiplier.
             * @return Shared pointer to the created ScalarMultEvaluator.
             */
            template <int DIM>
            typename ScalarMultEvaluator<DIM>::Ptr smult(
                const typename Evaluable<typename ScalarMultEvaluator<DIM>::InType>::ConstPtr& v,
                const double& s);

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam
