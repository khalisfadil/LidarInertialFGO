#pragma once

#include <Eigen/Core>
#include <memory>

#include "LGMath/LieGroupMath.hpp"
#include "Core/Evaluable/Evaluable.hpp"
#include "Core/Trajectory/ConstAcceleration/Variables.hpp"
#include "Core/Trajectory/Time.hpp"


namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            /**
             * @class PoseInterpolator
             * @brief Interpolates **SE(3) poses** between two trajectory knots in a constant-acceleration motion model.
             *
             * This interpolator estimates the pose at a given **interpolation time**, maintaining smooth trajectory transitions.
             */
            class PoseInterpolator : public slam::eval::Evaluable<slam::liemath::se3::Transformation> {
            public:

                using Ptr = std::shared_ptr<PoseInterpolator>;
                using ConstPtr = std::shared_ptr<const PoseInterpolator>;

                using InPoseType = slam::liemath::se3::Transformation;
                using InVelType = Eigen::Matrix<double, 6, 1>;
                using InAccType = Eigen::Matrix<double, 6, 1>;
                using OutType = slam::liemath::se3::Transformation;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create a shared pointer to `PoseInterpolator`.
                 *
                 * @param time The time at which interpolation is performed.
                 * @param knot1 First (earlier) knot.
                 * @param knot2 Second (later) knot.
                 * @return Shared pointer to the newly created `PoseInterpolator` instance.
                 */
                static Ptr MakeShared(const Time& time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs a `PoseInterpolator` for interpolating SE(3) poses.
                 *
                 * @param time The time at which interpolation is performed.
                 * @param knot1 First (earlier) knot.
                 * @param knot2 Second (later) knot.
                 */
                PoseInterpolator(const Time& time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Determines whether this interpolator is active in optimization.
                 * @return `true` if at least one of the dependent variables is active.
                 */
                bool active() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves the set of variable keys related to this interpolator.
                 * @param[out] keys Set of related variable keys.
                 */
                void getRelatedVarKeys(KeySet& keys) const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes the interpolated SE(3) pose.
                 * @return Interpolated transformation.
                 */
                OutType value() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Performs forward evaluation, computing the interpolated pose.
                 * @return A node containing the computed transformation.
                 */
                slam::eval::Node<OutType>::Ptr forward() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Performs backward propagation to accumulate Jacobians.
                 *
                 * @param lhs Left-hand side of the Jacobian product.
                 * @param node The node containing the computed transformation.
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

                /** @brief Interpolation values */
                Eigen::Matrix<double, 18, 18> omega_;
                Eigen::Matrix<double, 18, 18> lambda_;
            };

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
