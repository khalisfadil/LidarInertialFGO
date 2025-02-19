#pragma once

#include <Eigen/Core>
#include "source/include/LGMath/LieGroupMath.hpp"
#include "source/include/Evaluable/Evaluable.hpp"
#include "source/include/Trajectory/ConstAcceleration/Variables.hpp"
#include "source/include/Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {
            
            // -----------------------------------------------------------------------------
            /**
             * @brief AccelerationInterpolator interpolates acceleration between two knots in a constant-acceleration trajectory.
             */
            class AccelerationInterpolator : public slam::eval::Evaluable<Eigen::Matrix<double, 6, 1>> {
                public:
                using Ptr = std::shared_ptr<AccelerationInterpolator>;
                using ConstPtr = std::shared_ptr<const AccelerationInterpolator>;

                using InPoseType = slam::liemath::se3::Transformation;
                using InVelType = Eigen::Matrix<double, 6, 1>;
                using InAccType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 6, 1>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create a shared pointer instance.
                 * @param time The time at which acceleration is interpolated.
                 * @param knot1 First (earlier) state variable.
                 * @param knot2 Second (later) state variable.
                 * @return Shared pointer to AccelerationInterpolator instance.
                 */
                static Ptr MakeShared(const Time time, const Variable::ConstPtr& knot1,
                                        const Variable::ConstPtr& knot2);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for AccelerationInterpolator.
                 * @param time The time at which acceleration is interpolated.
                 * @param knot1 First (earlier) state variable.
                 * @param knot2 Second (later) state variable.
                 */
                AccelerationInterpolator(const Time time, const Variable::ConstPtr& knot1,
                                        const Variable::ConstPtr& knot2);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Check if this evaluator depends on active variables.
                 * @return True if dependent on active variables, false otherwise.
                 */
                bool active() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieve the set of related variable keys.
                 * @param keys Set to be populated with variable keys.
                 */
                void getRelatedVarKeys(KeySet& keys) const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Compute the interpolated acceleration value.
                 * @return Interpolated acceleration as a 6x1 vector.
                 */
                OutType value() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Perform forward evaluation for factor graph optimization.
                 * @return Shared pointer to the resulting node containing the interpolated acceleration.
                 */
                slam::eval::Node<OutType>::Ptr forward() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Compute the Jacobians during backpropagation.
                 * @param lhs Left-hand side matrix for Jacobian accumulation.
                 * @param node Resulting node containing the interpolated acceleration.
                 * @param jacs Jacobians storage structure to be populated.
                 */
                void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs, 
                                const slam::eval::Node<OutType>::Ptr& node,
                                slam::eval::StateKeyJacobians& jacs) const override;

                // -----------------------------------------------------------------------------
                /** @brief Omega matrix used for interpolation calculations. */
                Eigen::Matrix<double, 18, 18> omega_;
                /** @brief Lambda matrix used for interpolation calculations. */
                Eigen::Matrix<double, 18, 18> lambda_;

            protected:

                // -----------------------------------------------------------------------------
                /** @brief First (earlier) knot in the trajectory. */
                const Variable::ConstPtr knot1_;

                // -----------------------------------------------------------------------------
                /** @brief Second (later) knot in the trajectory. */
                const Variable::ConstPtr knot2_;
            };

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
