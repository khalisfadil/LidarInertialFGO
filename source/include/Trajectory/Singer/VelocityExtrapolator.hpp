#pragma once

#include <Eigen/Core>

#include "source/include/Trajectory/ConstAcceleration/Variables.hpp"
#include "source/include/Trajectory/ConstAcceleration/VelocityExtrapolator.hpp"
#include "source/include/Trajectory/Singer/Helper.hpp"
#include "source/include/Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace singer {
            
            // -----------------------------------------------------------------------------
            /**
             * @brief Factory method to create a shared instance of `PoseInterpolator`.
             *
             * @param time  Interpolation time at which the pose is estimated.
             * @param knot1 First trajectory knot (state) providing pose, velocity, and acceleration.
             * @param knot2 Second trajectory knot (state) defining motion continuity.
             * @param ad    Damping coefficient vector (6x1) affecting acceleration dynamics.
             * @return      Shared pointer to a `PoseInterpolator` instance.
             */
            class VelocityExtrapolator : public slam::traj::const_acc::VelocityExtrapolator {
                public:
                    using Ptr = std::shared_ptr<VelocityExtrapolator>;
                    using ConstPtr = std::shared_ptr<const VelocityExtrapolator>;
                    using Variable = slam::traj::const_acc::Variable;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared instance of `PoseInterpolator`.
                     *
                     * @param time  Interpolation time at which the pose is estimated.
                     * @param knot1 First trajectory knot (state) providing pose, velocity, and acceleration.
                     * @param knot2 Second trajectory knot (state) defining motion continuity.
                     * @param ad    Damping coefficient vector (6x1) affecting acceleration dynamics.
                     * @return      Shared pointer to a `PoseInterpolator` instance.
                     */
                    static Ptr MakeShared(const Time& time, const Variable::ConstPtr& knot,
                                            const Eigen::Matrix<double, 6, 1>& ad);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared instance of `PoseInterpolator`.
                     *
                     * @param time  Interpolation time at which the pose is estimated.
                     * @param knot1 First trajectory knot (state) providing pose, velocity, and acceleration.
                     * @param knot2 Second trajectory knot (state) defining motion continuity.
                     * @param ad    Damping coefficient vector (6x1) affecting acceleration dynamics.
                     * @return      Shared pointer to a `PoseInterpolator` instance.
                     */
                    VelocityExtrapolator(const Time& time, const Variable::ConstPtr& knot,
                                            const Eigen::Matrix<double, 6, 1>& ad);
            };
        }  // namespace singer
    }  // namespace traj
}  // namespace slam
