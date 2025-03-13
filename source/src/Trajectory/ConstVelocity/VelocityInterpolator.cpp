#include "Trajectory/ConstAcceleration/VelocityInterpolator.hpp"
#include "Trajectory/ConstAcceleration/Helper.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            /** 
             * @brief Creates a shared pointer to a VelocityInterpolator instance.
             * @param time The interpolation time.
             * @param knot1 Pointer to the first knot variable.
             * @param knot2 Pointer to the second knot variable.
             * @return Shared pointer to the created VelocityInterpolator.
             */
            auto VelocityInterpolator::MakeShared(
                const Time& time, const Variable::ConstPtr& knot1,
                const Variable::ConstPtr& knot2) -> Ptr {
                return std::make_shared<VelocityInterpolator>(time, knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // Constructor
            // -----------------------------------------------------------------------------

            /**
             * @brief Constructs a VelocityInterpolator between two knots at a given time.
             * @param time The interpolation time.
             * @param knot1 Pointer to the first knot variable.
             * @param knot2 Pointer to the second knot variable.
             * @throws std::invalid_argument If time intervals are invalid.
             */
            VelocityInterpolator::VelocityInterpolator(const Time& time,
                                           const Variable::ConstPtr& knot1,
                                           const Variable::ConstPtr& knot2)
                : knot1_(knot1), knot2_(knot2) {
                // Compute time constants
                const double T = (knot2->getTime() - knot1->getTime()).seconds();
                const double tau = (time - knot1->getTime()).seconds();
                const double kappa = (knot2->getTime() - time).seconds();

                // Validate time intervals
                if (T <= 0) {
                    throw std::invalid_argument("Total time T must be positive");
                }
                if (tau < 0 || kappa < 0) {
                    throw std::invalid_argument("Interpolation time must be between knots");
                }

                // Precompute shared values
                static const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
                const auto Qinv_T = getQinv(T, ones);
                const auto Tran_T = getTran(T);

                // Compute interpolation values efficiently
                omega_ = getQ(tau, ones) * getTran(kappa).transpose() * Qinv_T;
                lambda_ = getTran(tau) - omega_ * Tran_T;
            }

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------

            /**
             * @brief Checks if any related variables are active.
             * @return True if any pose, velocity, or acceleration is active, false otherwise.
             */
            bool VelocityInterpolator::active() const {
                return knot1_->getPose()->active() || knot1_->getVelocity()->active() ||
                    knot1_->getAcceleration()->active() || knot2_->getPose()->active() ||
                    knot2_->getVelocity()->active() || knot2_->getAcceleration()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            /**
             * @brief Retrieves keys of related variables.
             * @param keys Set to store the related variable keys.
             */
            void VelocityInterpolator::getRelatedVarKeys(eval::Evaluable<InPoseType>::KeySet& keys) const {
                knot1_->getPose()->getRelatedVarKeys(keys);
                knot1_->getVelocity()->getRelatedVarKeys(keys);
                knot1_->getAcceleration()->getRelatedVarKeys(keys);
                knot2_->getPose()->getRelatedVarKeys(keys);
                knot2_->getVelocity()->getRelatedVarKeys(keys);
                knot2_->getAcceleration()->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            /**
             * @brief Computes the interpolated velocity value.
             * @return The interpolated velocity as an OutType.
             */
            auto VelocityInterpolator::value() const -> OutType {
                // Retrieve state values
                const auto T1 = knot1_->getPose()->value(), T2 = knot2_->getPose()->value();
                const auto w1 = knot1_->getVelocity()->value(), dw1 = knot1_->getAcceleration()->value();
                const auto w2 = knot2_->getVelocity()->value(), dw2 = knot2_->getAcceleration()->value();

                // Compute se(3) transformation and Jacobian
                const auto xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);
                const auto Jw2 = J_21_inv * w2, Jdw2 = J_21_inv * dw2;

                // Split omega_12 computation
                const auto curlyhat_Jw2 = liemath::se3::curlyhat(Jw2);
                const auto temp_omega_12 = curlyhat_Jw2 * w2;
                const auto omega_12 = -0.5 * temp_omega_12 + Jdw2;

                // Compute interpolated xi values using lambda_
                auto compute_xi = [&](int row_offset) -> Eigen::Matrix<double, 6, 1> {
                    return lambda_.block<6, 6>(row_offset, 6) * w1 +
                           lambda_.block<6, 6>(row_offset, 12) * dw1 +
                           omega_.block<6, 6>(row_offset, 0) * xi_21 +
                           omega_.block<6, 6>(row_offset, 6) * Jw2 +
                           omega_.block<6, 6>(row_offset, 12) * omega_12;
                };

                const Eigen::Matrix<double, 6, 1> xi_i1 = compute_xi(0);
                const Eigen::Matrix<double, 6, 1> xi_j1 = compute_xi(6);

                // Compute interpolated velocity
                return liemath::se3::vec2jac(xi_i1) * xi_j1;
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            /**
             * @brief Computes the forward pass for automatic differentiation.
             * @return A shared pointer to the evaluation node with the interpolated velocity.
             */
            auto VelocityInterpolator::forward() const -> slam::eval::Node<OutType>::Ptr {
                // Retrieve forward state values
                const auto T1 = knot1_->getPose()->forward(), T2 = knot2_->getPose()->forward();
                const auto w1 = knot1_->getVelocity()->forward(), dw1 = knot1_->getAcceleration()->forward();
                const auto w2 = knot2_->getVelocity()->forward(), dw2 = knot2_->getAcceleration()->forward();

                // Compute se(3) transformation and Jacobian
                const auto xi_21 = (T2->value() / T1->value()).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);
                const auto Jw2 = J_21_inv * w2->value(), Jdw2 = J_21_inv * dw2->value();

                // Split omega_12 computation
                const auto curlyhat_Jw2 = liemath::se3::curlyhat(Jw2);
                const auto temp_omega_12 = curlyhat_Jw2 * w2->value();
                const auto omega_12 = -0.5 * temp_omega_12 + Jdw2;

                // Lambda function to compute xi values
                auto compute_xi = [&](int row_offset) -> Eigen::Matrix<double, 6, 1> {
                    return lambda_.block<6, 6>(row_offset, 6) * w1->value() +
                           lambda_.block<6, 6>(row_offset, 12) * dw1->value() +
                           omega_.block<6, 6>(row_offset, 0) * xi_21 +
                           omega_.block<6, 6>(row_offset, 6) * Jw2 +
                           omega_.block<6, 6>(row_offset, 12) * omega_12;
                };

                // Compute interpolated values
                const Eigen::Matrix<double, 6, 1> xi_i1 = compute_xi(0);
                const Eigen::Matrix<double, 6, 1> xi_j1 = compute_xi(6);

                // Compute interpolated velocity
                const auto node = slam::eval::Node<OutType>::MakeShared(liemath::se3::vec2jac(xi_i1) * xi_j1);

                // Explicitly specify the container type
                std::initializer_list<slam::eval::NodeBase::Ptr> children = {T1, w1, dw1, T2, w2, dw2};
                for (const auto& child : children) node->addChild(child);
                
                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            /**
             * @brief Computes the backward pass for Jacobian propagation.
             * @param lhs Left-hand side matrix for chain rule.
             * @param node Evaluation node from the forward pass.
             * @param jacs Container for storing Jacobians.
             */
            void VelocityInterpolator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                                const eval::Node<OutType>::Ptr& node,
                                                eval::StateKeyJacobians& jacs) const {
                if (!active()) return;

                // Retrieve state values
                const auto T1 = knot1_->getPose()->value(), T2 = knot2_->getPose()->value();
                const auto w1 = knot1_->getVelocity()->value(), dw1 = knot1_->getAcceleration()->value();
                const auto w2 = knot2_->getVelocity()->value(), dw2 = knot2_->getAcceleration()->value();

                // Compute se(3) transformation and Jacobian
                const auto xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);
                const auto Jw2 = J_21_inv * w2, Jdw2 = J_21_inv * dw2;
                const auto curlyhat_Jw2 = liemath::se3::curlyhat(Jw2);
                const auto curlyhat_w2 = liemath::se3::curlyhat(w2);
                const auto curlyhat_dw2 = liemath::se3::curlyhat(dw2);
                const auto temp_omega_12 = curlyhat_Jw2 * w2;
                const auto omega_12 = -0.5 * temp_omega_12 + Jdw2;

                // Compute interpolated xi values using lambda_
                auto compute_xi = [&](int row_offset) -> Eigen::Matrix<double, 6, 1> {
                    return lambda_.block<6, 6>(row_offset, 6) * w1 +
                           lambda_.block<6, 6>(row_offset, 12) * dw1 +
                           omega_.block<6, 6>(row_offset, 0) * xi_21 +
                           omega_.block<6, 6>(row_offset, 6) * Jw2 +
                           omega_.block<6, 6>(row_offset, 12) * omega_12;
                };

                const Eigen::Matrix<double, 6, 1> xi_i1 = compute_xi(0);
                const Eigen::Matrix<double, 6, 1> xi_j1 = compute_xi(6);

                // Compute transformation and Jacobians
                const liemath::se3::Transformation T_21(xi_21, 0);
                const Eigen::Matrix<double, 6, 6> J_i1 = liemath::se3::vec2jac(xi_i1);
                const auto xi_j1_ch = -0.5 * liemath::se3::curlyhat(xi_j1);

                // Precompute common Jacobian expressions with intermediates
                const auto omega_66 = omega_.block<6, 6>(6, 6);
                const auto omega_612 = omega_.block<6, 6>(6, 12);
                const auto half_curlyhat_w2 = 0.5 * curlyhat_w2;
                const auto term1 = omega_66 * half_curlyhat_w2;
                const auto curlyhat_w2_sq = curlyhat_w2 * curlyhat_w2;
                const auto quarter_curlyhat_w2_sq = 0.25 * curlyhat_w2_sq;
                const auto term2 = omega_612 * quarter_curlyhat_w2_sq;
                const auto half_curlyhat_dw2 = 0.5 * curlyhat_dw2;
                const auto term3 = omega_612 * half_curlyhat_dw2;
                const auto J_prep_intermediate = omega_.block<6, 6>(6, 0) + term1 + term2 + term3;
                const auto J_prep = J_i1 * J_prep_intermediate * J_21_inv;

                // Update pose Jacobians
                if (knot1_->getPose()->active() || knot2_->getPose()->active()) {
                    if (knot1_->getPose()->active()) {
                        const auto T1_ = std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(0));
                        const auto T21_adjoint = T_21.adjoint();
                        const auto J_prep_T21 = J_prep * T21_adjoint;
                        const auto neg_J_prep_T21 = -J_prep_T21;
                        const auto lhs_neg_J_prep_T21 = lhs * neg_J_prep_T21;
                        knot1_->getPose()->backward(lhs_neg_J_prep_T21, T1_, jacs);
                    }
                    if (knot2_->getPose()->active()) {
                        const auto T2_ = std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(3));
                        const auto lhs_J_prep = lhs * J_prep;
                        knot2_->getPose()->backward(lhs_J_prep, T2_, jacs);
                    }
                }

                // Process Jacobians for velocity and acceleration with intermediates
                std::array<std::function<void()>, 4> jacobian_updates = {
                    [&] {
                        if (knot1_->getVelocity()->active()) {
                            const auto lambda_66 = lambda_.block<6, 6>(6, 6);
                            const auto lambda_06 = lambda_.block<6, 6>(0, 6);
                            const auto J_i1_lambda_66 = J_i1 * lambda_66;
                            const auto xi_j1_ch_lambda_06 = xi_j1_ch * lambda_06;
                            const auto term_vel1 = J_i1_lambda_66 + xi_j1_ch_lambda_06;
                            const auto lhs_term_vel1 = lhs * term_vel1;
                            knot1_->getVelocity()->backward(lhs_term_vel1, 
                                std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(1)), jacs);
                        }
                    },
                    [&] {
                        if (knot2_->getVelocity()->active()) {
                            const auto omega_66_J21 = omega_66 * J_21_inv;
                            const auto curlyhat_diff = curlyhat_Jw2 - curlyhat_w2 * J_21_inv;
                            const auto neg_half_curlyhat_diff = -0.5 * curlyhat_diff;
                            const auto omega_612_term = omega_612 * neg_half_curlyhat_diff;
                            const auto term_vel2_part1 = omega_66_J21 + omega_612_term;
                            const auto omega_06_J21 = omega_.block<6, 6>(0, 6) * J_21_inv;
                            const auto omega_012_term = omega_.block<6, 6>(0, 12) * neg_half_curlyhat_diff;
                            const auto term_vel2_part2 = omega_06_J21 + omega_012_term;
                            const auto J_i1_part1 = J_i1 * term_vel2_part1;
                            const auto xi_j1_ch_part2 = xi_j1_ch * term_vel2_part2;
                            const auto term_vel2 = J_i1_part1 + xi_j1_ch_part2;
                            const auto lhs_term_vel2 = lhs * term_vel2;
                            knot2_->getVelocity()->backward(lhs_term_vel2, 
                                std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(4)), jacs);
                        }
                    },
                    [&] {
                        if (knot1_->getAcceleration()->active()) {
                            const auto lambda_612 = lambda_.block<6, 6>(6, 12);
                            const auto lambda_012 = lambda_.block<6, 6>(0, 12);
                            const auto J_i1_lambda_612 = J_i1 * lambda_612;
                            const auto xi_j1_ch_lambda_012 = xi_j1_ch * lambda_012;
                            const auto term_acc1 = J_i1_lambda_612 + xi_j1_ch_lambda_012;
                            const auto lhs_term_acc1 = lhs * term_acc1;
                            knot1_->getAcceleration()->backward(lhs_term_acc1, 
                                std::static_pointer_cast<slam::eval::Node<InAccType>>(node->getChild(2)), jacs);
                        }
                    },
                    [&] {
                        if (knot2_->getAcceleration()->active()) {
                            const auto omega_612_J21 = omega_612 * J_21_inv;
                            const auto omega_012_J21 = omega_.block<6, 6>(0, 12) * J_21_inv;
                            const auto J_i1_omega_612_J21 = J_i1 * omega_612_J21;
                            const auto xi_j1_ch_omega_012_J21 = xi_j1_ch * omega_012_J21;
                            const auto term_acc2 = J_i1_omega_612_J21 + xi_j1_ch_omega_012_J21;
                            const auto lhs_term_acc2 = lhs * term_acc2;
                            knot2_->getAcceleration()->backward(lhs_term_acc2, 
                                std::static_pointer_cast<slam::eval::Node<InAccType>>(node->getChild(5)), jacs);
                        }
                    }
                };

                for (const auto& update : jacobian_updates) update();
            }

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam