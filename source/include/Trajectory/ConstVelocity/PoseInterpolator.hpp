#pragma once

#include <Eigen/Core>
#include <memory>

#include "LGMath/LieGroupMath.hpp"
#include "Evaluable/Evaluable.hpp"
#include "Trajectory/ConstVelocity/Variables.hpp"
#include "Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {

            // -----------------------------------------------------------------------------
            /**
             * @class PoseInterpolator
             * @brief Interpolates pose using a constant velocity motion model.
             *
             * This class estimates an interpolated pose at a given timestamp
             * using two trajectory knots, assuming constant velocity motion.
             */
            class PoseInterpolator : public slam::eval::Evaluable<slam::liemath::se3::Transformation> {
            public:
                using Ptr = std::shared_ptr<PoseInterpolator>;
                using ConstPtr = std::shared_ptr<const PoseInterpolator>;

                using InPoseType = slam::liemath::se3::Transformation;
                using InVelType = Eigen::Matrix<double, 6, 1>;
                using OutType = slam::liemath::se3::Transformation;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create a shared instance of PoseInterpolator.
                 * @param time The time at which to interpolate the pose.
                 * @param knot1 The earlier control point.
                 * @param knot2 The later control point.
                 * @return Shared pointer to the created PoseInterpolator instance.
                 */
                static Ptr MakeShared(const slam::traj::Time& time,
                                      const Variable::ConstPtr& knot1,
                                      const Variable::ConstPtr& knot2);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs a PoseInterpolator.
                 * @param time The time at which to interpolate the pose.
                 * @param knot1 The earlier control point.
                 * @param knot2 The later control point.
                 */
                explicit PoseInterpolator(const slam::traj::Time& time,
                                          const Variable::ConstPtr& knot1,
                                          const Variable::ConstPtr& knot2);

                // -----------------------------------------------------------------------------
                /** @brief Checks if the interpolator is active (i.e., depends on active state variables). */
                bool active() const override;

                // -----------------------------------------------------------------------------
                /** @brief Retrieves keys of related variables. */
                void getRelatedVarKeys(KeySet& keys) const override;

                // -----------------------------------------------------------------------------
                /** @brief Computes the interpolated pose. */
                OutType value() const override;

                // -----------------------------------------------------------------------------
                /** @brief Computes the forward evaluation of pose interpolation. */
                slam::eval::Node<OutType>::Ptr forward() const override;

                // -----------------------------------------------------------------------------
                /** @brief Computes the backward propagation of Jacobians. */
                void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs, 
                              const slam::eval::Node<OutType>::Ptr& node, 
                              slam::eval::StateKeyJacobians& jacs) const override;

            private:
                /** @brief First (earlier) knot */
                const Variable::ConstPtr knot1_;

                /** @brief Second (later) knot */
                const Variable::ConstPtr knot2_;

                /** @brief Interpolation coefficients */
                double psi11_, psi12_, psi21_, psi22_;
                double lambda11_, lambda12_, lambda21_, lambda22_;
            };

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
