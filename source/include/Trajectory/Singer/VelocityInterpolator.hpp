#pragma once

#include <Eigen/Core>
#include <memory>

#include "source/include/Trajectory/ConstAcceleration/Variables.hpp"
#include "source/include/Trajectory/ConstAcceleration/VelocityInterpolator.hpp"
#include "source/include/Trajectory/Singer/Helper.hpp"
#include "source/include/Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace singer {
            
            // -----------------------------------------------------------------------------
            /**
             * @class VelocityInterpolator
             * @brief Estimates **velocity interpolation** between two trajectory knots  
             *        under a **damped motion model** in **Lie Algebra (se(3))**.
             *
             * **Key Features:**
             * - Computes **continuous-time velocity estimates**.
             * - Uses **damping coefficients** for motion control.
             * - Supports **Gaussian Process (GP) priors**.
             */
            class VelocityInterpolator : public slam::traj::const_acc::VelocityInterpolator {
            public:
                using Ptr = std::shared_ptr<VelocityInterpolator>;
                using ConstPtr = std::shared_ptr<const VelocityInterpolator>;
                using Variable = slam::traj::const_acc::Variable;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create a `VelocityInterpolator` instance.
                 *
                 * @param time  The query time at which velocity is interpolated.
                 * @param knot1 The first (earlier) trajectory knot.
                 * @param knot2 The second (later) trajectory knot.
                 * @param ad    Damping coefficient vector (6x1) affecting acceleration dynamics.
                 * @return      Shared pointer to `VelocityInterpolator`.
                 */
                static Ptr MakeShared(const Time& time, 
                                      const Variable::ConstPtr& knot1,
                                      const Variable::ConstPtr& knot2,
                                      const Eigen::Matrix<double, 6, 1>& ad);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs the `VelocityInterpolator` and precomputes interpolation matrices.
                 *
                 * @param time  The query time at which velocity is interpolated.
                 * @param knot1 The first (earlier) trajectory knot.
                 * @param knot2 The second (later) trajectory knot.
                 * @param ad    Damping coefficient vector (6x1) affecting acceleration dynamics.
                 */
                VelocityInterpolator(const Time& time, 
                                     const Variable::ConstPtr& knot1,
                                     const Variable::ConstPtr& knot2,
                                     const Eigen::Matrix<double, 6, 1>& ad);
            };

        }  // namespace singer
    }  // namespace traj
}  // namespace slam
