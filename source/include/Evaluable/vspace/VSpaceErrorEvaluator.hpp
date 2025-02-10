#pragma once

#include <memory>
#include <Eigen/Core>

#include "source/include/Evaluable/Evaluable.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // -----------------------------------------------------------------------------
            /**
             * @class VSpaceErrorEvaluator
             * @brief Computes the difference (error) between a vector-valued function and a measured value.
             *
             * **Functionality:**
             * - Computes **v_meas - v**, where `v_meas` is a fixed measurement and `v` is an evaluable function.
             * - Supports forward and backward propagation for **automatic differentiation**.
             *
             * @tparam DIM Dimensionality of the input vector (can be dynamic).
             */
            template <int DIM = Eigen::Dynamic>
            class VSpaceErrorEvaluator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
                public:
                    using Ptr = std::shared_ptr<VSpaceErrorEvaluator>;
                    using ConstPtr = std::shared_ptr<const VSpaceErrorEvaluator>;

                    using InType = Eigen::Matrix<double, DIM, 1>;
                    using OutType = Eigen::Matrix<double, DIM, 1>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method for creating a shared instance.
                     * @param v Input vector function.
                     * @param v_meas Measured vector value.
                     */
                    static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v, const InType& v_meas);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor for VSpaceErrorEvaluator.
                     * @param v Input vector function.
                     * @param v_meas Measured vector value.
                     */
                    VSpaceErrorEvaluator(const typename Evaluable<InType>::ConstPtr& v, const InType& v_meas);

                    // -----------------------------------------------------------------------------
                    /** @brief Checks if this evaluator depends on active state variables. */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Retrieves keys of related state variables. */
                    void getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Computes the error (difference) between the measured value and the function output. */
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
                    const InType v_meas_;  ///< Fixed measured vector value.
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Convenience function for creating a VSpaceErrorEvaluator instance.
             * @param v Input vector function.
             * @param v_meas Measured vector value.
             * @return Shared pointer to the created VSpaceErrorEvaluator.
             */
            template <int DIM>
            typename VSpaceErrorEvaluator<DIM>::Ptr vspace_error(
                const typename Evaluable<typename VSpaceErrorEvaluator<DIM>::InType>::ConstPtr& v,
                const typename VSpaceErrorEvaluator<DIM>::InType& v_meas);

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam
