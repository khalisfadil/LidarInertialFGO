#pragma once

#include <Eigen/Core>
#include <memory>

#include "LGMath/LieGroupMath.hpp"
#include "Core/Evaluable/Evaluable.hpp"
#include "Core/Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {

            // -----------------------------------------------------------------------------
            /**
             * @class Variable
             * @brief Represents a constant velocity motion model state.
             *
             * This class stores a timestamped **pose and velocity** to define
             * the trajectory evolution over time.
             */
            class Variable {
            public:
                using Ptr = std::shared_ptr<Variable>;
                using ConstPtr = std::shared_ptr<const Variable>;

                using PoseType = slam::liemath::se3::Transformation;
                using VelocityType = Eigen::Matrix<double, 6, 1>;

                // -----------------------------------------------------------------------------
                /** @brief Factory method to create a shared instance of Variable. */
                static Ptr MakeShared(const slam::traj::Time& time,
                                      const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                                      const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink);

                /** @brief Constructs a Variable. */
                explicit Variable(const slam::traj::Time& time,
                                  const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                                  const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink);

                /** @brief Default destructor */
                ~Variable() = default;

                // -----------------------------------------------------------------------------
                /** @brief Get the timestamp of this variable. */
                const slam::traj::Time& getTime() const;

                // -----------------------------------------------------------------------------
                /** @brief Get the pose evaluator. */
                const slam::eval::Evaluable<PoseType>::Ptr& getPose() const;

                // -----------------------------------------------------------------------------
                /** @brief Get the velocity evaluator. */
                const slam::eval::Evaluable<VelocityType>::Ptr& getVelocity() const;

            private:
                /** @brief Timestamp of the variable */
                slam::traj::Time time_;

                /** @brief Pose evaluator (SE(3) transformation) */
                slam::eval::Evaluable<PoseType>::Ptr T_k0_;

                /** @brief Velocity evaluator (Lie algebra velocity) */
                slam::eval::Evaluable<VelocityType>::Ptr w_0k_ink_;
            };

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
