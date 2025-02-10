#pragma once

#include <memory>
#include <Eigen/Core>

#include "source/include/Evaluable/Evaluable.hpp"
#include "source/include/Evaluable/Node.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // -----------------------------------------------------------------------------
            /**
             * @class NegationEvaluator
             * @brief Computes the negation of a vector-valued function.
             *
             * **Functionality:**
             * - Computes **-v** where `v` is a vector-valued function.
             * - Supports **forward propagation** (computing values).
             * - Supports **backward propagation** (Jacobian computation).
             * - Tracks dependencies for **automatic differentiation**.
             *
             * @tparam DIM Dimensionality of input vector (can be dynamic).
             */
            template <int DIM = Eigen::Dynamic>
            class NegationEvaluator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
                public:
                    using Ptr = std::shared_ptr<NegationEvaluator>;
                    using ConstPtr = std::shared_ptr<const NegationEvaluator>;

                    using InType = Eigen::Matrix<double, DIM, 1>;
                    using OutType = Eigen::Matrix<double, DIM, 1>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method for creating a shared instance.
                     * @param v Function to be negated.
                     */
                    static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor for NegationEvaluator.
                     * @param v Function to be negated.
                     */
                    explicit NegationEvaluator(const typename Evaluable<InType>::ConstPtr& v);

                    // -----------------------------------------------------------------------------
                    /** @brief Determines if this evaluator depends on active variables. */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Retrieves keys of related state variables. */
                    void getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Computes the negated output value. */
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
                    const typename Evaluable<InType>::ConstPtr v_;  ///< Input function.
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Convenience function for creating a NegationEvaluator.
             * @param v Function to be negated.
             */
            template <int DIM>
            typename NegationEvaluator<DIM>::Ptr neg(
                const typename Evaluable<typename NegationEvaluator<DIM>::InType>::ConstPtr& v);

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam
