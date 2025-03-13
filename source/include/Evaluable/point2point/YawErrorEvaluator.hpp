#pragma once

#include <Eigen/Core>

#include "LGMath/LieGroupMath.hpp"
#include "Evaluable/Evaluable.hpp"

namespace slam {
    namespace eval {
        namespace p2p {

            // -----------------------------------------------------------------------------
            /**
             * @class YawErrorEvaluator
             * @brief Evaluates the yaw error for factor graph optimization.
             *
             * Computes yaw error given a measured yaw \( \psi_{meas} \) and an estimated yaw state.
             * Used for yaw-based constraints in SLAM and sensor fusion.
             */
            class YawErrorEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
            public:
                using Ptr = std::shared_ptr<YawErrorEvaluator>;
                using ConstPtr = std::shared_ptr<const YawErrorEvaluator>;
                using PoseInType = liemath::se3::Transformation;
                using OutType = Eigen::Matrix<double, 1, 1>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create an instance.
                 * @param yaw_meas Measured yaw.
                 * @param T_ms_prev Evaluated transformation at previous timestamp.
                 * @param T_ms_curr Evaluated transformation at current timestamp.
                 * @return Shared pointer to a new evaluator instance.
                 */
                static Ptr MakeShared(double yaw_meas, 
                                    const Evaluable<PoseInType>::ConstPtr &T_ms_prev, 
                                    const Evaluable<PoseInType>::ConstPtr &T_ms_curr);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor.
                 * @param yaw_meas Measured yaw.
                 * @param T_ms_prev Evaluated transformation at previous timestamp.
                 * @param T_ms_curr Evaluated transformation at current timestamp.
                 */
                YawErrorEvaluator(double yaw_meas, 
                                const Evaluable<PoseInType>::ConstPtr &T_ms_prev, 
                                const Evaluable<PoseInType>::ConstPtr &T_ms_curr);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Checks if the yaw error is influenced by active state variables.
                 * @return True if active, otherwise false.
                 */
                bool active() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Collects state variable keys that influence this evaluator.
                 * @param[out] keys Set of related state keys.
                 */
                void getRelatedVarKeys(KeySet& keys) const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes the yaw error.
                 * @return Yaw error as a 1x1 matrix.
                 */
                OutType value() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Forward evaluation of yaw error.
                 * @return Shared pointer to the computed yaw error node.
                 */
                Node<OutType>::Ptr forward() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes Jacobians for the yaw error.
                 * @param lhs Left-hand-side weight matrix.
                 * @param node Transformation estimate node.
                 * @param jacs Container for accumulating Jacobians.
                 */
                void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                            const Node<OutType>::Ptr& node,
                            StateKeyJacobians& jacs) const override;

            private:
                const double yaw_meas_;  ///< Measured yaw angle.
                const Evaluable<PoseInType>::ConstPtr T_ms_prev_;  ///< Transformation at previous state.
                const Evaluable<PoseInType>::ConstPtr T_ms_curr_;  ///< Transformation at current state.

                Eigen::Matrix<double, 1, 3> d_;  ///< Jacobian storage.
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory function for creating a yaw error evaluator.
             * @param yaw_meas Measured yaw.
             * @param T_ms_prev Evaluated transformation at previous timestamp.
             * @param T_ms_curr Evaluated transformation at current timestamp.
             * @return Shared pointer to the evaluator.
             */
            YawErrorEvaluator::Ptr yawError(double yaw_meas, 
                                            const Evaluable<YawErrorEvaluator::PoseInType>::ConstPtr &T_ms_prev,
                                            const Evaluable<YawErrorEvaluator::PoseInType>::ConstPtr &T_ms_curr);

        }  // namespace p2p
    }  // namespace eval
}  // namespace slam
