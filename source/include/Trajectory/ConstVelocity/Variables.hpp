#pragma once

#include <Eigen/Core>
#include <memory>

#include "source/include/LGMath/LieGroupMath.hpp"
#include "source/include/Evaluable/Evaluable.hpp"
#include "source/include/Trajectory/Time.hpp"

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
                /**
                 * @brief Factory method to create a shared instance of Variable.
                 * @param time Timestamp of the variable.
                 * @param T_k0 Pose evaluator (transformation from keyframe `k` to `0`).
                 * @param w_0k_ink Velocity evaluator (velocity of `k` in frame `0`).
                 * @return Shared pointer to created Variable instance.
                 */
                static Ptr MakeShared(const slam::traj::Time& time,
                                      const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                                      const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink) {
                    return std::make_shared<Variable>(time, T_k0, w_0k_ink);
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs a Variable.
                 * @param time Timestamp of the variable.
                 * @param T_k0 Pose evaluator (SE(3) transformation).
                 * @param w_0k_ink Velocity evaluator (Lie algebra velocity).
                 */
                explicit Variable(const slam::traj::Time& time,
                                  const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                                  const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink)
                    : time_(time), T_k0_(std::move(T_k0)), w_0k_ink_(std::move(w_0k_ink)) {}

                /** @brief Default destructor */
                ~Variable() = default;

                // -----------------------------------------------------------------------------
                /** @brief Get the timestamp of this variable. */
                const slam::traj::Time& getTime() const { return time_; }

                // -----------------------------------------------------------------------------
                /** @brief Get the pose evaluator. */
                const slam::eval::Evaluable<PoseType>::Ptr& getPose() const { return T_k0_; }

                // -----------------------------------------------------------------------------
                /** @brief Get the velocity evaluator. */
                const slam::eval::Evaluable<VelocityType>::Ptr& getVelocity() const { return w_0k_ink_; }

            private:
                // -----------------------------------------------------------------------------
                /** @brief Timestamp of the variable */
                slam::traj::Time time_;

                // -----------------------------------------------------------------------------
                /** @brief Pose evaluator (SE(3) transformation) */
                slam::eval::Evaluable<PoseType>::Ptr T_k0_;

                // -----------------------------------------------------------------------------
                /** @brief Velocity evaluator (Lie algebra velocity) */
                slam::eval::Evaluable<VelocityType>::Ptr w_0k_ink_;
            };

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
