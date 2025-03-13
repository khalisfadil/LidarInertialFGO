#include "Trajectory/ConstAcceleration/VelocityInterpolator.hpp"
#include "Trajectory/ConstAcceleration/Helper.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto VelocityInterpolator::MakeShared(
                const Time& time, const Variable::ConstPtr& knot1,
                const Variable::ConstPtr& knot2) -> Ptr {
                return std::make_shared<VelocityInterpolator>(time, knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // Constructor
            // -----------------------------------------------------------------------------

           VelocityInterpolator::VelocityInterpolator(const Time& time,
                                           const Variable::ConstPtr& knot1,
                                           const Variable::ConstPtr& knot2)
                : knot1_(knot1), knot2_(knot2) {
                // Compute time constants
                const double T = (knot2->getTime() - knot1->getTime()).seconds();
                const double tau = (time - knot1->getTime()).seconds();
                const double kappa = (knot2->getTime() - time).seconds();

                // Validate time intervals
                if (T <= 0) throw std::invalid_argument("Total time T must be positive");
                if (tau < 0 || kappa < 0) throw std::invalid_argument("Time must be between knots");

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

            bool VelocityInterpolator::active() const {
                return knot1_->getPose()->active() || knot1_->getVelocity()->active() ||
                    knot1_->getAcceleration()->active() || knot2_->getPose()->active() ||
                    knot2_->getVelocity()->active() || knot2_->getAcceleration()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

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

            auto VelocityInterpolator::value() const -> OutType {
                // Retrieve state values
                const auto T1 = knot1_->getPose()->value(), T2 = knot2_->getPose()->value();
                const auto w1 = knot1_->getVelocity()->value(), dw1 = knot1_->getAcceleration()->value();
                const auto w2 = knot2_->getVelocity()->value(), dw2 = knot2_->getAcceleration()->value();

                // Compute se(3) transformation and Jacobian
                const auto xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);
                const auto Jw2 = J_21_inv * w2, Jdw2 = J_21_inv * dw2;

                // Split omega_12 computation for clarity and Eigen compatibility
                const auto curlyhat_Jw2 = liemath::se3::curlyhat(Jw2);
                const auto temp = curlyhat_Jw2 * w2;
                const auto omega_12 = -0.5 * temp + Jdw2;

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
                const auto temp = curlyhat_Jw2 * w2->value();
                const auto omega_12 = -0.5 * temp + Jdw2;

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

                // Split omega_12 computation
                const auto curlyhat_Jw2 = liemath::se3::curlyhat(Jw2);
                const auto temp = curlyhat_Jw2 * w2;
                const auto omega_12 = -0.5 * temp + Jdw2;

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

                // Precompute J_prep with intermediate steps
                const auto curlyhat_w2 = liemath::se3::curlyhat(w2);
                const auto curlyhat_dw2 = liemath::se3::curlyhat(dw2);
                const auto curlyhat_w2_sq = curlyhat_w2 * curlyhat_w2;
                const auto term1 = omega_.block<6, 6>(6, 0);
                const auto term2 = omega_.block<6, 6>(6, 6) * (0.5 * curlyhat_w2);
                const auto term3 = omega_.block<6, 6>(6, 12) * (0.25 * curlyhat_w2_sq);
                const auto term4 = omega_.block<6, 6>(6, 12) * (0.5 * curlyhat_dw2);
                const auto sum_terms = term1 + term2 + term3 + term4;
                const auto J_prep = J_i1 * sum_terms * J_21_inv;

                // Compute pose Jacobians with intermediate negation
                if (knot1_->getPose()->active() || knot2_->getPose()->active()) {
                    if (knot1_->getPose()->active()) {
                        const auto T1_ = std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(0));
                        const auto neg_J_prep = -J_prep;
                        const auto neg_J_prep_T21 = neg_J_prep * T_21.adjoint();
                        knot1_->getPose()->backward(lhs * neg_J_prep_T21, T1_, jacs);
                    }
                    if (knot2_->getPose()->active()) {
                        const auto T2_ = std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(3));
                        knot2_->getPose()->backward(lhs * J_prep, T2_, jacs);
                    }
                }

                // Process Jacobians using a lambda-based approach
                std::array<std::function<void()>, 4> jacobian_updates = {
                    [&] {
                        if (knot1_->getVelocity()->active())
                            knot1_->getVelocity()->backward(
                                lhs * (J_i1 * lambda_.block<6, 6>(6, 6) + xi_j1_ch * lambda_.block<6, 6>(0, 6)),
                                std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(1)), jacs);
                    },
                    [&] {
                        if (knot2_->getVelocity()->active())
                            knot2_->getVelocity()->backward(
                                lhs * (J_i1 * (omega_.block<6, 6>(6, 6) * J_21_inv +
                                            omega_.block<6, 6>(6, 12) * -0.5 * (curlyhat_Jw2 - curlyhat_w2 * J_21_inv)) +
                                    xi_j1_ch * (omega_.block<6, 6>(0, 6) * J_21_inv +
                                                omega_.block<6, 6>(0, 12) * -0.5 * (curlyhat_Jw2 - curlyhat_w2 * J_21_inv))),
                                std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(4)), jacs);
                    },
                    [&] {
                        if (knot1_->getAcceleration()->active())
                            knot1_->getAcceleration()->backward(
                                lhs * (J_i1 * lambda_.block<6, 6>(6, 12) + xi_j1_ch * lambda_.block<6, 6>(0, 12)),
                                std::static_pointer_cast<slam::eval::Node<InAccType>>(node->getChild(2)), jacs);
                    },
                    [&] {
                        if (knot2_->getAcceleration()->active())
                            knot2_->getAcceleration()->backward(
                                lhs * (J_i1 * omega_.block<6, 6>(6, 12) * J_21_inv +
                                    xi_j1_ch * omega_.block<6, 6>(0, 12) * J_21_inv),
                                std::static_pointer_cast<slam::eval::Node<InAccType>>(node->getChild(5)), jacs);
                    }
                };

                for (const auto& update : jacobian_updates) update();
            }
        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
