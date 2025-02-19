#pragma once

#include <Eigen/Core>
#include <memory>

#include "source/include/Evaluable/Evaluable.hpp"
#include "source/include/Trajectory/ConstAcceleration/Variables.hpp"
#include "source/include/Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {
            
            // -----------------------------------------------------------------------------
            /**
             * @class VelocityExtrapolator
             * @brief Extrapolates velocity from a known knot in a constant acceleration trajectory.
             *
             * Given a trajectory knot at a known time, this class estimates the velocity
             * at a future time by using the **constant acceleration model**.
             */
            class VelocityExtrapolator : public eval::Evaluable<Eigen::Matrix<double, 6, 1>> {
                public:
                    using Ptr = std::shared_ptr<VelocityExtrapolator>;
                    using ConstPtr = std::shared_ptr<const VelocityExtrapolator>;

                    using InVelType = Eigen::Matrix<double, 6, 1>;
                    using InAccType = Eigen::Matrix<double, 6, 1>;
                    using OutType = Eigen::Matrix<double, 6, 1>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared pointer instance.
                     * @param time The extrapolation time.
                     * @param knot The known trajectory knot.
                     * @return Shared pointer to the new VelocityExtrapolator.
                     */
                    static Ptr MakeShared(const Time time, const Variable::ConstPtr& knot);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor to initialize extrapolator with known state.
                     * @param time The extrapolation time.
                     * @param knot The known trajectory knot.
                     */
                    VelocityExtrapolator(const Time time, const Variable::ConstPtr& knot);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Checks if any associated variables are active.
                     * @return True if any associated variable is active.
                     */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Retrieves the set of related variable keys.
                     * @param keys The set to store related variable keys.
                     */
                    void getRelatedVarKeys(eval::Evaluable<OutType>::KeySet& keys) const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the extrapolated velocity value.
                     * @return The extrapolated velocity as a 6x1 vector.
                     */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Forward pass to create a computational node for evaluation.
                     * @return Shared pointer to a node containing the computed velocity.
                     */
                    slam::eval::Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Backward pass to compute and accumulate Jacobians.
                     * @param lhs The left-hand-side matrix.
                     * @param node The computational node.
                     * @param jacs The container for Jacobians.
                     */
                    void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                    const eval::Node<OutType>::Ptr& node,
                                    slam::eval::StateKeyJacobians& jacs) const override;

                protected:
                    // -----------------------------------------------------------------------------
                    /** @brief Knot from which velocity is extrapolated */
                    const Variable::ConstPtr knot_;

                    // -----------------------------------------------------------------------------
                    /** @brief Transition matrix for constant acceleration extrapolation */
                    Eigen::Matrix<double, 18, 18> Phi_;
            };

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
