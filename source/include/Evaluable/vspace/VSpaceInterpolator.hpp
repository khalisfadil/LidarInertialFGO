#pragma once

#include <memory>
#include <Eigen/Core>

#include "Evaluable/Evaluable.hpp"
#include "Trajectory/Time.hpp"

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
                                      const Time& time2) {
                    return std::make_shared<VSpaceInterpolator>(time, bias1, time1, bias2, time2);
                }

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
                                   const Time& time2)
                    : bias1_(bias1), bias2_(bias2) {
                    // Ensure the interpolation time is within valid range
                    if (time < time1 || time > time2) {
                        throw std::runtime_error("[VSpaceInterpolator] Interpolation time out of range.");
                    }

                    // Compute interpolation weights
                    const double tau = (time - time1).seconds();
                    const double T = (time2 - time1).seconds();
                    const double ratio = tau / T;
                    
                    psi_ = ratio;
                    lambda_ = 1.0 - ratio;
                }

                // -----------------------------------------------------------------------------
                /** @brief Checks if this evaluator depends on active state variables. */
                bool active() const override {
                    return bias1_->active() || bias2_->active();
                }

                // -----------------------------------------------------------------------------
                /** @brief Retrieves keys of related state variables. */
                void getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const override {
                    bias1_->getRelatedVarKeys(keys);
                    bias2_->getRelatedVarKeys(keys);
                }

                // -----------------------------------------------------------------------------
                /** @brief Computes the interpolated value. */
                OutType value() const override {
                    return lambda_ * bias1_->value() + psi_ * bias2_->value();
                }

                // -----------------------------------------------------------------------------
                /** @brief Creates a computation node for forward propagation. */
                typename Node<OutType>::Ptr forward() const override {
                    const auto b1 = bias1_->forward();
                    const auto b2 = bias2_->forward();

                    OutType interpolated_value = lambda_ * b1->value() + psi_ * b2->value();

                    const auto node = Node<OutType>::MakeShared(interpolated_value);
                    node->addChild(b1);
                    node->addChild(b2);
                    
                    return node;
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Performs backpropagation to accumulate Jacobians.
                 * @param lhs Left-hand-side weight matrix.
                 * @param node Computation node from `forward()`.
                 * @param jacs Storage for accumulated Jacobians.
                 */
                void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                              const typename Node<OutType>::Ptr& node,
                              StateKeyJacobians& jacs) const override {
                    if (!active()) return;

                    if (bias1_->active()) {
                        const auto b1_ = std::static_pointer_cast<Node<InType>>(node->getChild(0));
                        bias1_->backward(lambda_ * lhs, b1_, jacs);
                    }

                    if (bias2_->active()) {
                        const auto b2_ = std::static_pointer_cast<Node<InType>>(node->getChild(1));
                        bias2_->backward(psi_ * lhs, b2_, jacs);
                    }
                }

            private:
                const typename Evaluable<InType>::ConstPtr bias1_;  ///< First bias state.
                const typename Evaluable<InType>::ConstPtr bias2_;  ///< Second bias state.

                double psi_;    ///< Interpolation weight for `bias2`.
                double lambda_; ///< Interpolation weight for `bias1`.
            };

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam