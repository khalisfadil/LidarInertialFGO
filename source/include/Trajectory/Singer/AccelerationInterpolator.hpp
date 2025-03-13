#pragma once

#include <Eigen/Core>
#include <memory>

#include "Trajectory/ConstAcceleration/Variables.hpp"
#include "Trajectory/ConstAcceleration/AccelerationInterpolator.hpp"
#include "Trajectory/Singer/Helper.hpp"
#include "Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace singer {
            
            // -----------------------------------------------------------------------------
            /**
             * @class AccelerationInterpolator
             * @brief Interpolates **acceleration** between two trajectory knots under a **damped motion model**.
             *
             * This interpolator estimates the **Lie algebra (se(3)) acceleration** at a given time  
             * using **Gaussian Process (GP) priors** while incorporating damping dynamics.
             *
             * **Key Features:**
             * - Computes **continuous-time acceleration estimates**.
             * - Utilizes **damping coefficients** to control smoothness.
             * - Supports **Lie Algebra se(3) transformations**.
             */
            class AccelerationInterpolator : public slam::traj::const_acc::AccelerationInterpolator {
            public:
                using Ptr = std::shared_ptr<AccelerationInterpolator>;
                using ConstPtr = std::shared_ptr<const AccelerationInterpolator>;
                using Variable = slam::traj::const_acc::Variable;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create a shared instance of `AccelerationInterpolator`.
                 *
                 * @param time  Interpolation time at which acceleration is estimated.
                 * @param knot1 First trajectory knot (state) providing pose, velocity, and acceleration.
                 * @param knot2 Second trajectory knot (state) defining motion continuity.
                 * @param ad    Damping coefficient vector (6x1) affecting acceleration dynamics.
                 * @return      Shared pointer to an `AccelerationInterpolator` instance.
                 */
                static Ptr MakeShared(const Time& time, 
                                      const Variable::ConstPtr& knot1,
                                      const Variable::ConstPtr& knot2,
                                      const Eigen::Matrix<double, 6, 1>& ad);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs the `AccelerationInterpolator` with precomputed interpolation values.
                 *
                 * @param time  Interpolation time at which acceleration is estimated.
                 * @param knot1 First trajectory knot (state) providing pose, velocity, and acceleration.
                 * @param knot2 Second trajectory knot (state) defining motion continuity.
                 * @param ad    Damping coefficient vector (6x1) affecting acceleration dynamics.
                 */
                AccelerationInterpolator(const Time& time, 
                                         const Variable::ConstPtr& knot1,
                                         const Variable::ConstPtr& knot2,
                                         const Eigen::Matrix<double, 6, 1>& ad);
            };

        }  // namespace singer
    }  // namespace traj
}  // namespace slam
