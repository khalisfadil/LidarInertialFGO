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
             * @class AdditionEvaluator
             * @brief Computes element-wise addition of two vector-valued functions.
             *
             * **Functionality:**
             * - Computes **v1 + v2** at runtime.
             * - Supports **forward propagation** (computing values).
             * - Supports **backward propagation** (Jacobian computation).
             * - Tracks dependencies for **automatic differentiation**.
             *
             * @tparam DIM Dimensionality of input vectors (can be dynamic).
             */
            template <int DIM = Eigen::Dynamic>
            class AdditionEvaluator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
                public:
                    using Ptr = std::shared_ptr<AdditionEvaluator>;
                    using ConstPtr = std::shared_ptr<const AdditionEvaluator>;

                    using InType = Eigen::Matrix<double, DIM, 1>;
                    using OutType = Eigen::Matrix<double, DIM, 1>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method for creating a shared instance.
                     * @param v1 First function to be added.
                     * @param v2 Second function to be added.
                     */
                    static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v1,
                                        const typename Evaluable<InType>::ConstPtr& v2);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor for AdditionEvaluator.
                     * @param v1 First function to be added.
                     * @param v2 Second function to be added.
                     */
                    AdditionEvaluator(const typename Evaluable<InType>::ConstPtr& v1,
                                    const typename Evaluable<InType>::ConstPtr& v2);

                    // -----------------------------------------------------------------------------
                    /** @brief Determines if this evaluator depends on active variables. */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Retrieves keys of related state variables. */
                    void getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Computes the summed output value. */
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
                    const typename Evaluable<InType>::ConstPtr v1_;  ///< First input function.
                    const typename Evaluable<InType>::ConstPtr v2_;  ///< Second input function.
                };

                // -----------------------------------------------------------------------------
                /**
                 * @brief Convenience function for creating an AdditionEvaluator.
                 * @param v1 First function to be added.
                 * @param v2 Second function to be added.
                 */
                template <int DIM>
                typename AdditionEvaluator<DIM>::Ptr add(
                    const typename Evaluable<typename AdditionEvaluator<DIM>::InType>::ConstPtr& v1,
                    const typename Evaluable<typename AdditionEvaluator<DIM>::InType>::ConstPtr& v2);

        } // namespace vspace
    }  // namespace eval
}  // namespace slam
