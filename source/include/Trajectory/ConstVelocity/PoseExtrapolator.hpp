#pragma once

#include <Eigen/Core>
#include <memory>

#include "LGMath/LieGroupMath.hpp"
#include "Evaluable/Evaluable.hpp"
#include "Trajectory/Time.hpp"
#include "Trajectory/ConstVelocity/Variables.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {

            // -----------------------------------------------------------------------------
            /**
             * @class PoseExtrapolator
             * @brief Extrapolates pose using a constant velocity motion model.
             *
             * This class computes an extrapolated pose at a given timestamp using a single
             * trajectory control point (`knot`) and a constant velocity assumption.
             */
            class PoseExtrapolator : public slam::eval::Evaluable<slam::liemath::se3::Transformation> {
            public:
                using Ptr = std::shared_ptr<PoseExtrapolator>;
                using ConstPtr = std::shared_ptr<const PoseExtrapolator>;

                using InPoseType = slam::liemath::se3::Transformation;
                using InVelType = Eigen::Matrix<double, 6, 1>;
                using OutType = slam::liemath::se3::Transformation;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create a shared instance of PoseExtrapolator.
                 * @param time Time at which the pose is extrapolated.
                 * @param knot Control point containing pose and velocity.
                 * @return Shared pointer to the created PoseExtrapolator instance.
                 */
                static Ptr MakeShared(const slam::traj::Time& time, const Variable::ConstPtr& knot);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for PoseExtrapolator.
                 * @param time Time at which the pose is extrapolated.
                 * @param knot Control point containing pose and velocity.
                 */
                explicit PoseExtrapolator(const slam::traj::Time& time, const Variable::ConstPtr& knot);

                // -----------------------------------------------------------------------------
                /** @brief Checks if the extrapolator is active (pose or velocity is active). */
                bool active() const override;

                // -----------------------------------------------------------------------------
                /** @brief Retrieves keys of related variables. */
                void getRelatedVarKeys(KeySet& keys) const override;

                // -----------------------------------------------------------------------------
                /** @brief Computes the extrapolated pose value. */
                OutType value() const override;

                // -----------------------------------------------------------------------------
                /** @brief Computes the forward evaluation of the extrapolated pose. */
                slam::eval::Node<OutType>::Ptr forward() const override;

                // -----------------------------------------------------------------------------
                /** @brief Computes the backward propagation of Jacobians. */
                void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                              const slam::eval::Node<OutType>::Ptr& node,
                              slam::eval::StateKeyJacobians& jacs) const override;

            private:
                // -----------------------------------------------------------------------------
                /** @brief Control point for extrapolation */
                const Variable::ConstPtr knot_;

                // -----------------------------------------------------------------------------
                /** @brief Transition matrix for extrapolation */
                Eigen::Matrix<double, 12, 12> Phi_;
            };

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
