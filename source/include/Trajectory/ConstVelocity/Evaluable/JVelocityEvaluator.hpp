#pragma once

#include <Eigen/Core>
#include <memory>

#include "source/include/LGMath/LieGroupMath.hpp"
#include "source/include/Evaluable/Evaluable.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {

        // -----------------------------------------------------------------------------
        /**
         * @class JVelocityEvaluator
         * @brief Evaluates the Jacobian of velocity with respect to SE(3) pose perturbations.
         *
         * This class computes the left Jacobian transformation of velocity, required for
         * constant-velocity motion models in SLAM.
         *
         * The evaluator provides:
         * - **Forward computation** for velocity transformation.
         * - **Backward computation** for Jacobian accumulation.
         * - **Support for factor graph optimization** via `Evaluable<T>`.
         */
        class JVelocityEvaluator : public eval::Evaluable<Eigen::Matrix<double, 6, 1>> {
        public:

        using Ptr = std::shared_ptr<JVelocityEvaluator>;
        using ConstPtr = std::shared_ptr<const JVelocityEvaluator>;

        using XiInType = Eigen::Matrix<double, 6, 1>;  ///< Input pose perturbation (ξ in se(3)).
        using VelInType = Eigen::Matrix<double, 6, 1>; ///< Input velocity (v in se(3)).
        using OutType = Eigen::Matrix<double, 6, 1>;  ///< Output transformed velocity.

        // -----------------------------------------------------------------------------
        /**
         * @brief Factory method to create a shared pointer to `JVelocityEvaluator`.
         *
         * @param xi Pose perturbation evaluable (SE(3)).
         * @param velocity Velocity evaluable.
         * @return Shared pointer to the new `JVelocityEvaluator` instance.
         */
        static Ptr MakeShared(const eval::Evaluable<XiInType>::ConstPtr& xi,
                                const eval::Evaluable<VelInType>::ConstPtr& velocity);

        // -----------------------------------------------------------------------------
        /**
         * @brief Constructs a `JVelocityEvaluator` for computing the Jacobian of velocity.
         *
         * @param xi Pose perturbation evaluable (SE(3)).
         * @param velocity Velocity evaluable.
         */
        JVelocityEvaluator(const eval::Evaluable<XiInType>::ConstPtr& xi,
                            const eval::Evaluable<VelInType>::ConstPtr& velocity);

        // -----------------------------------------------------------------------------
        /**
         * @brief Checks whether this evaluator depends on active state variables.
         *
         * @return `true` if either `xi_` or `velocity_` are active, `false` otherwise.
         */
        bool active() const override;

        // -----------------------------------------------------------------------------
        /**
         * @brief Retrieves all state keys that influence this evaluator.
         *
         * @param keys The set of related state keys to be updated.
         */
        void getRelatedVarKeys(eval::Evaluable<XiInType>::KeySet& keys) const override;

        // -----------------------------------------------------------------------------
        /**
         * @brief Computes the velocity transformation using the left Jacobian.
         *
         * Computes:
         * \f[
         * J(\xi) v
         * \f]
         * where \( J(\xi) \) is the left Jacobian of SE(3).
         *
         * @return The transformed velocity.
         */
        OutType value() const override;

        // -----------------------------------------------------------------------------
        /**
         * @brief Performs the forward pass and constructs a computational node.
         *
         * - Retrieves the forward-evaluated values of `xi_` and `velocity_`.
         * - Computes the transformed velocity using `value()`.
         * - Stores the result in a `Node<OutType>`.
         *
         * @return Shared pointer to a `Node<OutType>` containing the computed velocity.
         */
        eval::Node<OutType>::Ptr forward() const override;

        // -----------------------------------------------------------------------------
        /**
         * @brief Performs the backward pass and accumulates Jacobians.
         *
         * Computes:
         * \f[
         * J_{\xi} = -0.5 \cdot lhs \cdot \text{curlyhat}(v)
         * \f]
         * for pose perturbation and:
         * \f[
         * J_v = lhs \cdot J(\xi)
         * \f]
         * for velocity.
         *
         * @param lhs Left-hand-side weight matrix from higher-level differentiation.
         * @param node Computational node from `forward()`.
         * @param jacs Jacobian storage container for accumulation.
         */
        void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                        const eval::Node<OutType>::Ptr& node,
                        eval::StateKeyJacobians& jacs) const override;

        private:
        
        const eval::Evaluable<XiInType>::ConstPtr xi_;  ///< Pose perturbation evaluable (SE(3)).
        const eval::Evaluable<VelInType>::ConstPtr velocity_;  ///< Velocity evaluable.

        };

        // -----------------------------------------------------------------------------
        /**
         * @brief Utility function to create a `JVelocityEvaluator` instance.
         *
         * Equivalent to calling `MakeShared()`, provided for convenience.
         *
         * @param xi Pose perturbation evaluable.
         * @param velocity Velocity evaluable.
         * @return Shared pointer to the new `JVelocityEvaluator`.
         */
        JVelocityEvaluator::Ptr j_velocity(
            const eval::Evaluable<JVelocityEvaluator::XiInType>::ConstPtr& xi,
            const eval::Evaluable<JVelocityEvaluator::VelInType>::ConstPtr& velocity);

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
