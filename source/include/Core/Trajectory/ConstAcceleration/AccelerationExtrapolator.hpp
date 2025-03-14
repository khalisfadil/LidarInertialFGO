#pragma once

#include <Eigen/Core>

#include "Core/Evaluable/Evaluable.hpp"
#include "Core/Trajectory/ConstAcceleration/Variables.hpp"
#include "Core/Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            /**
             * @class AccelerationExtrapolator
             * @brief Extrapolates acceleration using SE(3) transformations.
             *
             * This class predicts the future acceleration state using a constant acceleration
             * motion model and a transition matrix.
             */
            class AccelerationExtrapolator : public slam::eval::Evaluable<Eigen::Matrix<double, 6, 1>> {

                public:
                    using Ptr = std::shared_ptr<AccelerationExtrapolator>;
                    using ConstPtr = std::shared_ptr<const AccelerationExtrapolator>;

                    using InAccType = Eigen::Matrix<double, 6, 1>;
                    using OutType = Eigen::Matrix<double, 6, 1>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create an instance of `AccelerationExtrapolator`.
                     * @param time The time at which extrapolation is performed.
                     * @param knot The state variable to extrapolate from.
                     * @return Shared pointer to the created instance.
                     */
                    static Ptr MakeShared(const Time& time, const Variable::ConstPtr& knot);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructs an `AccelerationExtrapolator` instance.
                     * @param time The time at which extrapolation is performed.
                     * @param knot The state variable to extrapolate from.
                     */
                    AccelerationExtrapolator(const Time& time, const Variable::ConstPtr& knot);

                    // -----------------------------------------------------------------------------
                    /** @brief Checks if the extrapolator depends on any active variables. */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Retrieves related variable keys for factor graph optimization. */
                    void getRelatedVarKeys(KeySet& keys) const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Computes the extrapolated acceleration value. */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Computes the forward extrapolation and returns it as a node. */
                    slam::eval::Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /** 
                     * @brief Computes the backward pass, accumulating Jacobians.
                     * @param lhs Left-hand side of the Jacobian computation.
                     * @param node Node representing this extrapolator.
                     * @param jacs Container for accumulated Jacobians.
                     */
                    void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs, 
                                const slam::eval::Node<OutType>::Ptr& node,
                                slam::eval::StateKeyJacobians& jacs) const override;

                protected:

                    // -----------------------------------------------------------------------------
                    /** @brief The knot (state) to extrapolate from. */
                    const Variable::ConstPtr knot_;

                    // -----------------------------------------------------------------------------
                    /** @brief Transition matrix for constant acceleration extrapolation. */
                    Eigen::Matrix<double, 18, 18> Phi_;
            };
        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam