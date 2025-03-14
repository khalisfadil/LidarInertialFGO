#include "Core/Trajectory/ConstAcceleration/AccelerationInterpolator.hpp"

#include "Core/Evaluable/se3/Evaluables.hpp"
#include "Core/Evaluable/vspace/Evaluables.hpp"
#include "Core/Trajectory/ConstAcceleration/Helper.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {
            
            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            AccelerationInterpolator::Ptr AccelerationInterpolator::MakeShared(
                const Time time, const Variable::ConstPtr& knot1,
                const Variable::ConstPtr& knot2) {
                return std::make_shared<AccelerationInterpolator>(time, knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // AccelerationInterpolator
            // -----------------------------------------------------------------------------

            AccelerationInterpolator::AccelerationInterpolator(
                const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2)
                : knot1_(knot1), knot2_(knot2) {

                // Compute time constants
                const double T = (knot2->getTime() - knot1->getTime()).seconds();
                const double tau = (time - knot1->getTime()).seconds();
                const double kappa = (knot2->getTime() - time).seconds();

                // Use Eigen::Matrix::Constant to avoid redundant allocation
                static const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Constant(1.0);

                // Precompute transition and covariance matrices
                const Eigen::Matrix<double, 18, 18> Qinv_T = getQinv(T, ones);
                const Eigen::Matrix<double, 18, 18> Tran_T = getTran(T);

                // Compute interpolation values
                omega_ = getQ(tau, ones) * getTran(kappa).transpose() * Qinv_T;
                lambda_ = getTran(tau) - omega_ * Tran_T;
            }

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------

            bool AccelerationInterpolator::active() const {
                return knot1_->getPose()->active() || knot1_->getVelocity()->active() ||
                    knot1_->getAcceleration()->active() || knot2_->getPose()->active() ||
                    knot2_->getVelocity()->active() || knot2_->getAcceleration()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void AccelerationInterpolator::getRelatedVarKeys(KeySet& keys) const {
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

            auto AccelerationInterpolator::value() const -> OutType {
                // Retrieve state values
                const auto T1 = knot1_->getPose()->value();
                const auto T2 = knot2_->getPose()->value();
                const auto w1 = knot1_->getVelocity()->value();
                const auto dw1 = knot1_->getAcceleration()->value();
                const auto w2 = knot2_->getVelocity()->value();
                const auto dw2 = knot2_->getAcceleration()->value();

                // Compute relative pose and Jacobian
                const Eigen::Matrix<double, 6, 1> xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Precompute reused terms
                const Eigen::Matrix<double, 6, 1> xi_21_Jw2 = J_21_inv * w2;
                const Eigen::Matrix<double, 6, 1> omega_12 = -0.5 * liemath::se3::curlyhat(xi_21_Jw2) * w2 + J_21_inv * dw2;

                // Lambda to compute xi values efficiently
                auto compute_xi = [&](int r) -> Eigen::Matrix<double, 6, 1> {
                    return lambda_.block<6, 6>(r, 6) * w1 +
                        lambda_.block<6, 6>(r, 12) * dw1 +
                        omega_.block<6, 6>(r, 0) * xi_21 +
                        omega_.block<6, 6>(r, 6) * xi_21_Jw2 +
                        omega_.block<6, 6>(r, 12) * omega_12;
                };

                // Compute interpolated values
                const Eigen::Matrix<double, 6, 1> xi_i1 = compute_xi(0);
                const Eigen::Matrix<double, 6, 1> xi_j1 = compute_xi(6);
                const Eigen::Matrix<double, 6, 1> xi_k1 = compute_xi(12);

                const Eigen::Matrix<double, 6, 6> J_i1 = liemath::se3::vec2jac(xi_i1);
                const Eigen::Matrix<double, 6, 1> w_i = J_i1 * xi_j1;
                const Eigen::Matrix<double, 6, 1> correction = 0.5 * liemath::se3::curlyhat(xi_j1) * w_i;

                return J_i1 * (xi_k1 + correction);
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto AccelerationInterpolator::forward() const -> slam::eval::Node<OutType>::Ptr {
                // Retrieve state values
                const auto T1 = knot1_->getPose()->forward();
                const auto w1 = knot1_->getVelocity()->forward();
                const auto dw1 = knot1_->getAcceleration()->forward();
                const auto T2 = knot2_->getPose()->forward();
                const auto w2 = knot2_->getVelocity()->forward();
                const auto dw2 = knot2_->getAcceleration()->forward();

                // Compute se(3) algebra of relative transformation and its Jacobian
                const Eigen::Matrix<double, 6, 1> xi_21 = (T2->value() / T1->value()).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = slam::liemath::se3::vec2jacinv(xi_21);

                // Precompute reused terms with explicit types
                const Eigen::Matrix<double, 6, 1> w1_val = w1->value();
                const Eigen::Matrix<double, 6, 1> dw1_val = dw1->value();
                const Eigen::Matrix<double, 6, 1> w2_val = w2->value();
                const Eigen::Matrix<double, 6, 1> dw2_val = dw2->value();
                const Eigen::Matrix<double, 6, 1> xi_21_Jw2 = J_21_inv * w2_val;
                const Eigen::Matrix<double, 6, 1> omega_12 = -0.5 * slam::liemath::se3::curlyhat(xi_21_Jw2) * w2_val + J_21_inv * dw2_val;

                // Compute xi values using lambda function with explicit return type
                auto compute_xi = [&](int r) -> Eigen::Matrix<double, 6, 1> {
                    return lambda_.block<6, 6>(r, 6) * w1_val + lambda_.block<6, 6>(r, 12) * dw1_val +
                        omega_.block<6, 6>(r, 0) * xi_21 + omega_.block<6, 6>(r, 6) * xi_21_Jw2 +
                        omega_.block<6, 6>(r, 12) * omega_12;
                };

                // Compute interpolated values with explicit types
                const Eigen::Matrix<double, 6, 6> J_i1 = slam::liemath::se3::vec2jac(compute_xi(0));
                const Eigen::Matrix<double, 6, 1> w_i = J_i1 * compute_xi(6);
                const Eigen::Matrix<double, 6, 1> xi_k1 = compute_xi(12);
                const Eigen::Matrix<double, 6, 1> correction = 0.5 * slam::liemath::se3::curlyhat(compute_xi(6)) * w_i;
                const OutType dw_i = J_i1 * (xi_k1 + correction);

                // Create node and add children efficiently
                const auto node = slam::eval::Node<OutType>::MakeShared(dw_i);
                std::initializer_list<slam::eval::NodeBase::Ptr> children = {T1, w1, dw1, T2, w2, dw2};
                for (const auto& child : children) {
                    node->addChild(child);
                }

                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void AccelerationInterpolator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs, 
                                        const slam::eval::Node<OutType>::Ptr& node,
                                        slam::eval::StateKeyJacobians& jacs) const {
                if (!active()) return;

                // Retrieve state values
                const auto T1 = knot1_->getPose()->value();
                const auto w1 = knot1_->getVelocity()->value();
                const auto dw1 = knot1_->getAcceleration()->value();
                const auto T2 = knot2_->getPose()->value();
                const auto w2 = knot2_->getVelocity()->value();
                const auto dw2 = knot2_->getAcceleration()->value();

                // Compute SE(3) algebra and Jacobian
                const Eigen::Matrix<double, 6, 1> xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Precompute reused terms
                const Eigen::Matrix<double, 6, 1> J_w2 = J_21_inv * w2;
                const Eigen::Matrix<double, 6, 1> omega_12 = -0.5 * liemath::se3::curlyhat(J_w2) * w2 + J_21_inv * dw2;

                // Lambda to compute xi values
                auto compute_xi = [&](int r) -> Eigen::Matrix<double, 6, 1> {
                    return lambda_.block<6, 6>(r, 6) * w1 + lambda_.block<6, 6>(r, 12) * dw1 +
                        omega_.block<6, 6>(r, 0) * xi_21 + omega_.block<6, 6>(r, 6) * J_w2 +
                        omega_.block<6, 6>(r, 12) * omega_12;
                };

                // Compute interpolated values
                const Eigen::Matrix<double, 6, 1> xi_i1 = compute_xi(0);
                const Eigen::Matrix<double, 6, 1> xi_j1 = compute_xi(6);
                const Eigen::Matrix<double, 6, 1> xi_k1 = compute_xi(12);

                const Eigen::Matrix<double, 6, 6> J_i1 = liemath::se3::vec2jac(xi_i1);
                const Eigen::Matrix<double, 6, 1> w_i = J_i1 * xi_j1;
                const Eigen::Matrix<double, 6, 6> curly_xi_j1 = liemath::se3::curlyhat(xi_j1);
                const Eigen::Matrix<double, 6, 1> correction = 0.5 * curly_xi_j1 * w_i;

                // Precompute common Jacobian expressions
                const Eigen::Matrix<double, 6, 6> J_prep_2 = J_i1 * (-0.5 * liemath::se3::curlyhat(w_i) + 0.5 * curly_xi_j1 * J_i1);
                const Eigen::Matrix<double, 6, 6> J_prep_3 = -0.25 * J_i1 * curly_xi_j1 * curly_xi_j1 -
                                                            0.5 * liemath::se3::curlyhat(xi_k1 + correction);

                // Precompute terms for Jacobians
                const Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();
                const Eigen::Matrix<double, 6, 6> curly_w2 = liemath::se3::curlyhat(w2);
                const Eigen::Matrix<double, 6, 6> curly_w2_sq = 0.25 * curly_w2 * curly_w2;
                const Eigen::Matrix<double, 6, 6> curly_dw2 = 0.5 * liemath::se3::curlyhat(dw2);
                const Eigen::Matrix<double, 6, 6> curly_diff = -0.5 * (liemath::se3::curlyhat(J_w2) - curly_w2 * J_21_inv);

                const liemath::se3::Transformation T_21(xi_21,0);
                const Eigen::Matrix<double, 6, 6> T_21_adj = T_21.adjoint();

                // Precompute common Jacobian components
                auto compute_jacobian = [&](int r0, int r6, int r12, bool use_omega, bool use_diff = false) -> Eigen::Matrix<double, 6, 6> {
                    const auto& matrix = use_omega ? omega_ : lambda_;
                    Eigen::Matrix<double, 6, 6> result = J_i1 * matrix.block<6, 6>(12, r12);
                    if (use_diff) result += J_i1 * matrix.block<6, 6>(12, r6) * J_21_inv;
                    else result = result * J_21_inv;
                    result += J_prep_2 * matrix.block<6, 6>(6, r12) * J_21_inv +
                            J_prep_3 * matrix.block<6, 6>(0, r12) * J_21_inv;
                    if (use_diff) result += (J_prep_2 * matrix.block<6, 6>(6, r6) + J_prep_3 * matrix.block<6, 6>(0, r6)) * J_21_inv;
                    return result;
                };

                const Eigen::Matrix<double, 6, 6> w = J_i1 * (omega_.block<6, 6>(12, 0) * I +
                                                            omega_.block<6, 6>(12, 6) * 0.5 * curly_w2 +
                                                            omega_.block<6, 6>(12, 12) * (curly_w2_sq + curly_dw2)) * J_21_inv +
                                                    J_prep_2 * (omega_.block<6, 6>(6, 0) * I +
                                                                omega_.block<6, 6>(6, 6) * 0.5 * curly_w2 +
                                                                omega_.block<6, 6>(6, 12) * (curly_w2_sq + curly_dw2)) * J_21_inv +
                                                    J_prep_3 * (omega_.block<6, 6>(0, 0) * I +
                                                                omega_.block<6, 6>(0, 6) * 0.5 * curly_w2 +
                                                                omega_.block<6, 6>(0, 12) * (curly_w2_sq + curly_dw2)) * J_21_inv;

                // Lambda-based Jacobian updates
                std::array<std::function<void()>, 6> updates = {
                    [&]() {
                        if (knot1_->getPose()->active()) {
                            const auto T1_ = std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(0));
                            knot1_->getPose()->backward(lhs * (-w * T_21_adj), T1_, jacs);
                        }
                    },
                    [&]() {
                        if (knot2_->getPose()->active()) {
                            const auto T2_ = std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(3));
                            knot2_->getPose()->backward(lhs * w, T2_, jacs);
                        }
                    },
                    [&]() {
                        if (knot1_->getVelocity()->active()) {
                            const auto w1_ = std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(1));
                            knot1_->getVelocity()->backward(lhs * compute_jacobian(0, 6, 6, false), w1_, jacs);
                        }
                    },
                    [&]() {
                        if (knot2_->getVelocity()->active()) {
                            const auto w2_ = std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(4));
                            knot2_->getVelocity()->backward(lhs * (compute_jacobian(0, 6, 12, true, true) + J_i1 * omega_.block<6, 6>(12, 12) * curly_diff), w2_, jacs);
                        }
                    },
                    [&]() {
                        if (knot1_->getAcceleration()->active()) {
                            const auto dw1_ = std::static_pointer_cast<slam::eval::Node<InAccType>>(node->getChild(2));
                            knot1_->getAcceleration()->backward(lhs * compute_jacobian(0, 12, 12, false), dw1_, jacs);
                        }
                    },
                    [&]() {
                        if (knot2_->getAcceleration()->active()) {
                            const auto dw2_ = std::static_pointer_cast<slam::eval::Node<InAccType>>(node->getChild(5));
                            knot2_->getAcceleration()->backward(lhs * compute_jacobian(0, 12, 12, true), dw2_, jacs);
                        }
                    }
                };

                // Execute updates
                for (const auto& update : updates) update();
            }

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
