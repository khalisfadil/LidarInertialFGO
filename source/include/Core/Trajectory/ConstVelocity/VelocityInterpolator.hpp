#pragma once

#include <Eigen/Core>
#include <memory>

#include "LGMath/LieGroupMath.hpp"
#include "Core/Evaluable/Evaluable.hpp"
#include "Core/Trajectory/ConstVelocity/Variables.hpp"
#include "Core/Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {

            // -----------------------------------------------------------------------------
            /**
             * @class VelocityInterpolator
             * @brief Interpolates velocity using a constant velocity motion model.
             *
             * Computes an interpolated velocity at a given timestamp between two control points
             * (knots). The interpolation is performed using Lie group operations.
             */
            class VelocityInterpolator : public slam::eval::Evaluable<Eigen::Matrix<double, 6, 1>> {
            public:
                using Ptr = std::shared_ptr<VelocityInterpolator>;
                using ConstPtr = std::shared_ptr<const VelocityInterpolator>;

                using InPoseType = slam::liemath::se3::Transformation;
                using InVelType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 6, 1>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create a shared instance of VelocityInterpolator.
                 * @param time Interpolation timestamp.
                 * @param knot1 First control point (earlier in time).
                 * @param knot2 Second control point (later in time).
                 * @return Shared pointer to the created VelocityInterpolator instance.
                 */
                static Ptr MakeShared(const slam::traj::Time& time,
                                      const slam::traj::const_vel::Variable::ConstPtr& knot1,
                                      const slam::traj::const_vel::Variable::ConstPtr& knot2);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for VelocityInterpolator.
                 * @param time Interpolation timestamp.
                 * @param knot1 First control point.
                 * @param knot2 Second control point.
                 */
                explicit VelocityInterpolator(const slam::traj::Time& time,
                                              const slam::traj::const_vel::Variable::ConstPtr& knot1,
                                              const slam::traj::const_vel::Variable::ConstPtr& knot2);

                // -----------------------------------------------------------------------------
                /** @brief Checks if the interpolator is active (any control point is active). */
                bool active() const override;

                // -----------------------------------------------------------------------------
                /** @brief Retrieves keys of related state variables. */
                void getRelatedVarKeys(KeySet& keys) const override;

                // -----------------------------------------------------------------------------
                /** @brief Computes the interpolated velocity value. */
                OutType value() const override;

                // -----------------------------------------------------------------------------
                /** @brief Computes the forward evaluation of velocity. */
                slam::eval::Node<OutType>::Ptr forward() const override;

                // -----------------------------------------------------------------------------
                /** @brief Computes the backward propagation of Jacobians. */
                void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs, 
                              const slam::eval::Node<OutType>::Ptr& node, 
                              slam::eval::StateKeyJacobians& jacs) const override;

            private:
                // -----------------------------------------------------------------------------
                /** @brief First (earlier) control point. */
                const slam::traj::const_vel::Variable::ConstPtr knot1_;

                // -----------------------------------------------------------------------------
                /** @brief Second (later) control point. */
                const slam::traj::const_vel::Variable::ConstPtr knot2_;

                // -----------------------------------------------------------------------------
                /** @brief Interpolation coefficients (computed at construction). */
                double psi11_, psi12_, psi21_, psi22_;
                double lambda11_, lambda12_, lambda21_, lambda22_;
            };

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
