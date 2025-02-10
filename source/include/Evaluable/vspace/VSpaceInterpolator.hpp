#pragma once

#include <memory>
#include <Eigen/Core>

#include "source/include/Evaluable/Evaluable.hpp"
#include "source/include/Trajectory/Time.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // -----------------------------------------------------------------------------
            /**
             * @class VSpaceInterpolator
             * @brief Performs **linear interpolation** between two vector state variables over time.
             *
             * **Functionality:**
             * - Interpolates a vector-valued function between `bias1` (at `time1`) and `bias2` (at `time2`).
             * - Supports forward and backward propagation for **automatic differentiation**.
             * - Handles time extrapolation with error checking.
             *
             * @tparam DIM Dimensionality of the input vector (can be dynamic).
             */
            template <int DIM = Eigen::Dynamic>
            class VSpaceInterpolator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
                public:
                    using Ptr = std::shared_ptr<VSpaceInterpolator>;
                    using ConstPtr = std::shared_ptr<const VSpaceInterpolator>;

                    using InType = Eigen::Matrix<double, DIM, 1>;
                    using OutType = Eigen::Matrix<double, DIM, 1>;
                    using Time = slam::traj::Time;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method for creating a shared instance.
                     * @param time Target interpolation time.
                     * @param bias1 First vector-valued function at `time1`.
                     * @param time1 Timestamp of `bias1`.
                     * @param bias2 Second vector-valued function at `time2`.
                     * @param time2 Timestamp of `bias2`.
                     */
                    static Ptr MakeShared(const Time& time, 
                                          const typename Evaluable<InType>::ConstPtr& bias1, 
                                          const Time& time1, 
                                          const typename Evaluable<InType>::ConstPtr& bias2, 
                                          const Time& time2);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor for VSpaceInterpolator.
                     * @param time Target interpolation time.
                     * @param bias1 First vector-valued function at `time1`.
                     * @param time1 Timestamp of `bias1`.
                     * @param bias2 Second vector-valued function at `time2`.
                     * @param time2 Timestamp of `bias2`.
                     * @throws std::runtime_error If `time` is outside the range `[time1, time2]`.
                     */
                    VSpaceInterpolator(const Time& time, 
                                       const typename Evaluable<InType>::ConstPtr& bias1, 
                                       const Time& time1, 
                                       const typename Evaluable<InType>::ConstPtr& bias2, 
                                       const Time& time2);

                    // -----------------------------------------------------------------------------
                    /** @brief Checks if this evaluator depends on active state variables. */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Retrieves keys of related state variables. */
                    void getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Computes the interpolated value. */
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
                    const typename Evaluable<InType>::ConstPtr bias1_;  ///< First bias state.
                    const typename Evaluable<InType>::ConstPtr bias2_;  ///< Second bias state.

                    double psi_;    ///< Interpolation weight for `bias2`.
                    double lambda_; ///< Interpolation weight for `bias1`.
            };

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam
