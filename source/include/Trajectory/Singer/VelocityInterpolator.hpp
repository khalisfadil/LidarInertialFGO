#pragma once

#include <Eigen/Core>
#include <memory>

#include "Trajectory/ConstAcceleration/Variables.hpp"
#include "Trajectory/ConstAcceleration/VelocityInterpolator.hpp"
#include "Trajectory/Singer/Helper.hpp"
#include "Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace singer {

        // -----------------------------------------------------------------------------
        // Constants for clarity (optional, could be defined elsewhere)
        constexpr int DOF = 6;  // Degrees of freedom for the system

        /**
         * @class VelocityInterpolator
         * @brief Estimates velocity interpolation between two trajectory knots
         *        under a damped motion model in Lie Algebra (se(3)).
         *
         * This class extends constant acceleration interpolation by incorporating
         * damping coefficients, suitable for the Singer model in SLAM applications.
         *
         * **Key Features:**
         * - Computes continuous-time velocity estimates.
         * - Uses damping coefficients for motion control (6x1 vector).
         * - Supports Gaussian Process (GP) priors for trajectory smoothing.
         */
        class VelocityInterpolator : public slam::traj::const_acc::VelocityInterpolator {
        public:
            using Ptr = std::shared_ptr<VelocityInterpolator>;
            using ConstPtr = std::shared_ptr<const VelocityInterpolator>;
            using Variable = slam::traj::const_acc::Variable;

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory method to create a VelocityInterpolator instance.
             *
             * @param time  Query time for velocity interpolation.
             * @param knot1 First (earlier) trajectory knot.
             * @param knot2 Second (later) trajectory knot.
             * @param ad    Damping coefficient vector (6x1) for acceleration dynamics.
             * @return      Shared pointer to a VelocityInterpolator instance.
             */
            static Ptr MakeShared(const Time& time,
                                const Variable::ConstPtr& knot1,
                                const Variable::ConstPtr& knot2,
                                const Eigen::Matrix<double, DOF, 1>& ad);

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs the VelocityInterpolator and precomputes interpolation matrices.
             *
             * The constructor assumes knot1.time <= time <= knot2.time. Implementations
             * should validate time intervals to ensure non-negative durations.
             *
             * @param time  Query time for velocity interpolation.
             * @param knot1 First (earlier) trajectory knot.
             * @param knot2 Second (later) trajectory knot.
             * @param ad    Damping coefficient vector (6x1) for acceleration dynamics.
             */
            VelocityInterpolator(const Time& time,
                                const Variable::ConstPtr& knot1,
                                const Variable::ConstPtr& knot2,
                                const Eigen::Matrix<double, DOF, 1>& ad);
        };

        }  // namespace singer
    }  // namespace traj
}  // namespace slam