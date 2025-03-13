#pragma once

#include <Eigen/Core>
#include <memory>

#include "LGMath/LieGroupMath.hpp"
#include "Evaluable/Evaluable.hpp"
#include "Trajectory/ConstAcceleration/Variables.hpp"
#include "Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

        // -----------------------------------------------------------------------------
        /**
         * @class VelocityInterpolator
         * @brief Interpolates velocity between two knots in a constant acceleration model.
         *
         * This class estimates the interpolated velocity **in SE(3) Lie algebra (se(3))**  
         * using the **pose, velocity, and acceleration** of two trajectory knots.  
         *
         * **Used in:**
         * - **Factor graph optimization**
         * - **Sensor fusion & trajectory estimation**
         * - **Continuous-time SLAM applications**
         */
        class VelocityInterpolator : public eval::Evaluable<Eigen::Matrix<double, 6, 1>> {
        public:

            using Ptr = std::shared_ptr<VelocityInterpolator>;
            using ConstPtr = std::shared_ptr<const VelocityInterpolator>;

            using InPoseType = liemath::se3::Transformation;  ///< SE(3) Pose.
            using InVelType = Eigen::Matrix<double, 6, 1>;  ///< se(3) Velocity.
            using InAccType = Eigen::Matrix<double, 6, 1>;  ///< se(3) Acceleration.
            using OutType = Eigen::Matrix<double, 6, 1>;  ///< Interpolated velocity.

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory method for creating `VelocityInterpolator` instances.
             * @param time The query time at which velocity is interpolated.
             * @param knot1 The first (earlier) trajectory knot.
             * @param knot2 The second (later) trajectory knot.
             * @return Shared pointer to `VelocityInterpolator`.
             */
            static Ptr MakeShared(const Time& time, 
                                  const Variable::ConstPtr& knot1,
                                  const Variable::ConstPtr& knot2);

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs the interpolator and precomputes interpolation matrices.
             * @param time The query time at which velocity is interpolated.
             * @param knot1 The first (earlier) trajectory knot.
             * @param knot2 The second (later) trajectory knot.
             */
            VelocityInterpolator(const Time& time, 
                                 const Variable::ConstPtr& knot1,
                                 const Variable::ConstPtr& knot2);

            // -----------------------------------------------------------------------------
            /**
             * @brief Checks if the interpolator depends on active variables.
             */
            bool active() const override;

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves the state keys affecting this function.
             */
            void getRelatedVarKeys(slam::eval::Evaluable<InPoseType>::KeySet& keys) const override;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the interpolated velocity.
             * @return Interpolated velocity (se(3)).
             */
            OutType value() const override;

            // -----------------------------------------------------------------------------
            /**
             * @brief Performs the forward pass and constructs a computation node.
             * @return Node containing the interpolated velocity.
             */
            eval::Node<OutType>::Ptr forward() const override;

            // -----------------------------------------------------------------------------
            /**
             * @brief Performs the backward pass for gradient computation.
             * @param lhs Left-hand-side weight matrix.
             * @param node The computation node from `forward()`.
             * @param jacs Container for storing computed Jacobians.
             */
            void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                          const slam::eval::Node<OutType>::Ptr& node,
                          slam::eval::StateKeyJacobians& jacs) const override;

        protected:
            const Variable::ConstPtr knot1_;  ///< First trajectory knot.
            const Variable::ConstPtr knot2_;  ///< Second trajectory knot.
            Eigen::Matrix<double, 18, 18> omega_;  ///< Precomputed interpolation matrix.
            Eigen::Matrix<double, 18, 18> lambda_; ///< Precomputed interpolation matrix.
        };

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
