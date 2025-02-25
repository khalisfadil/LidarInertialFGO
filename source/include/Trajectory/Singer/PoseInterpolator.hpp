#pragma once

#include <Eigen/Core>
#include <memory>

#include "source/include/Trajectory/ConstAcceleration/PoseInterpolator.hpp"
#include "source/include/Trajectory/ConstAcceleration/Variables.hpp"
#include "source/include/Trajectory/Singer/Helper.hpp"
#include "source/include/Trajectory/Time.hpp"


namespace slam {
    namespace traj {
        namespace singer {
            
            // -----------------------------------------------------------------------------
            /**
             * @class PoseInterpolator
             * @brief Interpolates **SE(3) poses** between two trajectory knots using a constant-acceleration motion model.
             *
             * This class estimates the pose at a given **interpolation time**, ensuring smooth trajectory transitions
             * by leveraging **Gaussian Process (GP) regression** and **Lie algebra representations** for accurate 
             * motion modeling. It accounts for damping effects and smoothly blends positional and rotational states.
             *
             * @details
             * Given two consecutive trajectory knots (`knot1`, `knot2`), this interpolator computes:
             * - The interpolated pose **T_i0** using **Lie group integration**.
             * - The motion evolution is governed by **state transition matrices** derived from GP regression.
             * - Supports automatic differentiation, **factor graph optimization**, and efficient **Jacobian computation**.
             *
             * @note This is a **key component of continuous-time SLAM and sensor fusion frameworks**, ensuring **state smoothness**.
             * 
             * Mathematical Formulation:
             * Using the SE(3) representation, we model the interpolated pose as:
             * \f[
             * T_{i0} = \exp(\xi_{i1}) T_1
             * \f]
             * where:
             * - \f$ \xi_{i1} \f$ is the interpolated relative motion using **Gaussian Process priors**.
             * - **Pose transition matrices** \f$ \Phi(dt, ad) \f$ ensure smooth dynamics.
             *
             * @see VelocityInterpolator, AccelerationInterpolator, getTran(), getJacKnot1(), getQ()
             */
            class PoseInterpolator : public slam::traj::const_acc::PoseInterpolator {
                public:
                    using Ptr = std::shared_ptr<PoseInterpolator>;
                    using ConstPtr = std::shared_ptr<const PoseInterpolator>;
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
                    static Ptr MakeShared(const Time& time, const Variable::ConstPtr& knot1,
                        const Variable::ConstPtr& knot2,
                        const Eigen::Matrix<double, 6, 1>& ad);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructs a `PoseInterpolator` for interpolating SE(3) poses.
                     *
                     * @param time  Interpolation time at which the pose is estimated.
                     * @param knot1 First trajectory knot (state) providing pose, velocity, and acceleration.
                     * @param knot2 Second trajectory knot (state) defining motion continuity.
                     * @param ad    Damping coefficient vector (6x1) affecting acceleration dynamics.
                     */
                    PoseInterpolator(const Time& time, const Variable::ConstPtr& knot1,
                        const Variable::ConstPtr& knot2,
                        const Eigen::Matrix<double, 6, 1>& ad);
            };
        }  // namespace singer
    }  // namespace traj
}  // namespace slam