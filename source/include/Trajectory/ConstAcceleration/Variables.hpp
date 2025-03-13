#pragma once

#include <Eigen/Core>
#include <memory>

#include "LGMath/LieGroupMath.hpp"
#include "Evaluable/Evaluable.hpp"
#include "Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

        // -----------------------------------------------------------------------------
        /**
         * @class Variable
         * @brief Represents a state variable in a constant-acceleration motion model.
         *
         * This class encapsulates a **pose**, **velocity**, and **acceleration** at a given time.
         * It is designed for use in **factor graph optimization**, **trajectory estimation**, 
         * and **sensor fusion** applications.
         *
         * The stored variables are:
         * - **Pose (SE(3))**: Represents the transformation from keyframe `k` to the world.
         * - **Velocity (se(3))**: Body-frame velocity.
         * - **Acceleration (se(3))**: Body-frame acceleration.
         */
        class Variable {
        public:

            using Ptr = std::shared_ptr<Variable>;
            using ConstPtr = std::shared_ptr<const Variable>;

            using PoseType = liemath::se3::Transformation; ///< SE(3) transformation (pose).
            using VelocityType = Eigen::Matrix<double, 6, 1>; ///< se(3) velocity (twist).
            using AccelerationType = Eigen::Matrix<double, 6, 1>; ///< se(3) acceleration.

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory method to create a shared pointer to `Variable`.
             *
             * @param time The timestamp associated with this variable.
             * @param T_k0 Pose evaluable (SE(3)).
             * @param w_0k_ink Velocity evaluable.
             * @param dw_0k_ink Acceleration evaluable.
             * @return Shared pointer to the newly created `Variable` instance.
             */
            static Ptr MakeShared(const Time& time, 
                                  const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                                  const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink,
                                  const slam::eval::Evaluable<AccelerationType>::Ptr& dw_0k_ink);

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs a `Variable` representing pose, velocity, and acceleration at a given time.
             *
             * @param time The timestamp associated with this variable.
             * @param T_k0 Pose evaluable (SE(3)).
             * @param w_0k_ink Velocity evaluable.
             * @param dw_0k_ink Acceleration evaluable.
             */
            Variable(const Time& time, 
                     const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                     const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink,
                     const slam::eval::Evaluable<AccelerationType>::Ptr& dw_0k_ink);

            // -----------------------------------------------------------------------------
            /**
             * @brief Virtual destructor.
             */
            virtual ~Variable() = default;

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves the timestamp associated with this state variable.
             * @return Reference to the `Time` object.
             */
            const Time& getTime() const;

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves the pose evaluable (SE(3) transformation).
             * @return Shared pointer to the pose evaluable.
             */
            const slam::eval::Evaluable<PoseType>::Ptr& getPose() const;

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves the velocity evaluable (se(3) twist).
             * @return Shared pointer to the velocity evaluable.
             */
            const slam::eval::Evaluable<VelocityType>::Ptr& getVelocity() const;

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves the acceleration evaluable (se(3) acceleration).
             * @return Shared pointer to the acceleration evaluable.
             */
            const slam::eval::Evaluable<AccelerationType>::Ptr& getAcceleration() const;

        private:
            Time time_;  ///< Timestamp associated with this state.
            slam::eval::Evaluable<PoseType>::Ptr T_k0_;  ///< Pose evaluable.
            slam::eval::Evaluable<VelocityType>::Ptr w_0k_ink_;  ///< Velocity evaluable.
            slam::eval::Evaluable<AccelerationType>::Ptr dw_0k_ink_;  ///< Acceleration evaluable.
        };

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
