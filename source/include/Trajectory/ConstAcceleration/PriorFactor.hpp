#pragma once

#include <Eigen/Core>
#include <memory>

#include "source/include/LGMath/LieGroupMath.hpp"
#include "source/include/Evaluable/Evaluable.hpp"
#include "source/include/Trajectory/ConstAcceleration/Variables.hpp"
#include "source/include/Trajectory/ConstAcceleration/Helper.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            /**
             * @class PriorFactor
             * @brief Represents a prior factor between two consecutive trajectory knots in a **constant-acceleration motion model**.
             *
             * This factor enforces smooth **pose, velocity, and acceleration** transitions across knots,
             * maintaining physical constraints in the factor graph optimization.
             */
            class PriorFactor : public slam::eval::Evaluable<Eigen::Matrix<double, 18, 1>> {
                public:

                    using Ptr = std::shared_ptr<PriorFactor>;
                    using ConstPtr = std::shared_ptr<const PriorFactor>;

                    using InPoseType = slam::liemath::se3::Transformation;
                    using InVelType = Eigen::Matrix<double, 6, 1>;
                    using InAccType = Eigen::Matrix<double, 6, 1>;
                    using OutType = Eigen::Matrix<double, 18, 1>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared pointer to `PriorFactor`.
                     *
                     * @param knot1 First (earlier) knot.
                     * @param knot2 Second (later) knot.
                     * @return Shared pointer to the newly created `PriorFactor` instance.
                     */
                    static Ptr MakeShared(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructs a `PriorFactor` between two trajectory knots.
                     *
                     * @param knot1 First (earlier) knot.
                     * @param knot2 Second (later) knot.
                     */
                    PriorFactor(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Determines whether this factor is active in optimization.
                     * @return `true` if at least one of the dependent variables is active.
                     */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Retrieves the set of variable keys related to this factor.
                     * @param[out] keys Set of related variable keys.
                     */
                    void getRelatedVarKeys(KeySet& keys) const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the residual error for the prior factor.
                     * @return 18x1 residual vector representing the constraint violation.
                     */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Performs forward evaluation, computing the residual.
                     * @return A node containing the computed residual.
                     */
                    eval::Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Performs backward propagation to accumulate Jacobians.
                     *
                     * @param lhs Left-hand side of the Jacobian product.
                     * @param node The node containing the computed residual.
                     * @param jacs Accumulator for Jacobians.
                     */
                    void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                const eval::Node<OutType>::Ptr& node,
                                eval::StateKeyJacobians& jacs) const override;

                protected:
                    /** @brief First (earlier) knot */
                    const Variable::ConstPtr knot1_;

                    /** @brief Second (later) knot */
                    const Variable::ConstPtr knot2_;

                    /** @brief Transition matrix */
                    Eigen::Matrix<double, 18, 18> Phi_;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the Jacobian with respect to the first knot.
                     * @return The 18x18 Jacobian matrix.
                     */
                    Eigen::Matrix<double, 18, 18> getJacKnot1_() const;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the Jacobian with respect to the second knot.
                     * @return The 18x18 Jacobian matrix.
                     */
                    Eigen::Matrix<double, 18, 18> getJacKnot2_() const;
            };

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
