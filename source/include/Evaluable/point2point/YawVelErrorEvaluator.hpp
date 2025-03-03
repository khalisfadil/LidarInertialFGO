#pragma once

#include <Eigen/Core>
#include "source/include/Evaluable/Evaluable.hpp"

namespace slam {
    namespace eval {
        namespace p2p {

            // -----------------------------------------------------------------------------
            /**
             * @class YawVelErrorEvaluator
             * @brief Evaluates yaw velocity error for factor graph optimization.
             *
             * Computes yaw velocity error given a measured yaw velocity \( \dot{\psi}_{meas} \) and an estimated velocity state.
             * Used in velocity-based constraints for SLAM and sensor fusion.
             */
            class YawVelErrorEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
            public:
                using Ptr = std::shared_ptr<YawVelErrorEvaluator>;
                using ConstPtr = std::shared_ptr<const YawVelErrorEvaluator>;
                using InType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 1, 1>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create an instance.
                 * @param vel_meas Measured yaw velocity.
                 * @param w_iv_inv Estimated velocity state.
                 * @return Shared pointer to a new evaluator instance.
                 */
                static Ptr MakeShared(const Eigen::Matrix<double, 1, 1>& vel_meas,
                                    const Evaluable<InType>::ConstPtr& w_iv_inv);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor.
                 * @param vel_meas Measured yaw velocity.
                 * @param w_iv_inv Estimated velocity state.
                 */
                YawVelErrorEvaluator(const Eigen::Matrix<double, 1, 1>& vel_meas,
                                    const Evaluable<InType>::ConstPtr& w_iv_inv);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Checks if the yaw velocity error is influenced by active state variables.
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
                 * @brief Computes the yaw velocity error.
                 * @return Yaw velocity error as a 1x1 matrix.
                 */
                OutType value() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Forward evaluation of yaw velocity error.
                 * @return Shared pointer to the computed yaw velocity error node.
                 */
                Node<OutType>::Ptr forward() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes Jacobians for the yaw velocity error.
                 * @param lhs Left-hand-side weight matrix.
                 * @param node Velocity state node.
                 * @param jacs Container for accumulating Jacobians.
                 */
                void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                            const Node<OutType>::Ptr& node,
                            StateKeyJacobians& jacs) const override;

            private:
                const Evaluable<InType>::ConstPtr w_iv_inv_;  ///< Evaluated velocity state.
                const Eigen::Matrix<double, 1, 1> vel_meas_;  ///< Measured yaw velocity.
                const Eigen::Matrix<double, 1, 6> D_; ///< Jacobian selector matrix for yaw velocity.
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory function for creating a yaw velocity error evaluator.
             * @param vel_meas Measured yaw velocity.
             * @param w_iv_inv Estimated velocity state.
             * @return Shared pointer to the evaluator.
             */
            auto velError(const Eigen::Matrix<double, 1, 1>& vel_meas,
                        const Evaluable<YawVelErrorEvaluator::InType>::ConstPtr& w_iv_inv) -> YawVelErrorEvaluator::Ptr;

        }  // namespace p2p
    }  // namespace eval
}  // namespace slam
