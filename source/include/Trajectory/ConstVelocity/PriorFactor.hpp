#pragma once

#include <Eigen/Core>
#include <memory>

#include "source/include/LGMath/LieGroupMath.hpp"
#include "source/include/Evaluable/Evaluable.hpp"
#include "source/include/Trajectory/ConstVelocity/Variables.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {

            // -----------------------------------------------------------------------------
            /**
             * @class PriorFactor
             * @brief Defines a motion prior between two trajectory states.
             *
             * This class represents a **prior factor** enforcing the expected motion model
             * between two control points (knots) in a **constant velocity** trajectory.
             * It encodes constraints on both the pose and velocity evolution over time.
             */
            class PriorFactor : public slam::eval::Evaluable<Eigen::Matrix<double, 12, 1>> {
            public:
                using Ptr = std::shared_ptr<PriorFactor>;
                using ConstPtr = std::shared_ptr<const PriorFactor>;

                using InPoseType = slam::liemath::se3::Transformation;
                using InVelType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 12, 1>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create a shared instance of PriorFactor.
                 * @param knot1 First trajectory control point.
                 * @param knot2 Second trajectory control point.
                 * @return Shared pointer to the created PriorFactor instance.
                 */
                static Ptr MakeShared(const Variable::ConstPtr& knot1,
                                      const Variable::ConstPtr& knot2) {
                    return std::make_shared<PriorFactor>(knot1, knot2);
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs a motion prior between two control points.
                 * @param knot1 First control point (earlier).
                 * @param knot2 Second control point (later).
                 */
                explicit PriorFactor(const Variable::ConstPtr& knot1,
                                     const Variable::ConstPtr& knot2);

                // -----------------------------------------------------------------------------
                /** @brief Checks if the prior factor depends on active variables. */
                bool active() const override;

                // -----------------------------------------------------------------------------
                /** @brief Retrieves keys of related state variables. */
                void getRelatedVarKeys(KeySet& keys) const override;

                // -----------------------------------------------------------------------------
                /** @brief Computes the prior factor residual. */
                OutType value() const override;

                // -----------------------------------------------------------------------------
                /** @brief Computes the forward evaluation of the prior factor. */
                slam::eval::Node<OutType>::Ptr forward() const override;

                // -----------------------------------------------------------------------------
                /** @brief Computes the backward propagation of Jacobians. */
                void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs, 
                              const slam::eval::Node<OutType>::Ptr& node, 
                              slam::eval::StateKeyJacobians& jacs) const override;

            private:
                // -----------------------------------------------------------------------------
                /** @brief First control point (earlier in time). */
                const Variable::ConstPtr knot1_;

                // -----------------------------------------------------------------------------
                /** @brief Second control point (later in time). */
                const Variable::ConstPtr knot2_;
            };

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
