#include "source/include/Trajectory/ConstAcceleration/AccelerationInterpolator.hpp"

#include "source/include/Evaluable/se3/Evaluables.hpp"
#include "source/include/Evaluable/vspace/Evaluables.hpp"
#include "source/include/Trajectory/ConstAcceleration/Helper.hpp"

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
                const auto xi_21 = (knot2_->getPose()->value() / knot1_->getPose()->value()).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = slam::liemath::se3::vec2jacinv(xi_21);

                // Precompute reused terms
                const auto w1 = knot1_->getVelocity()->value(), dw1 = knot1_->getAcceleration()->value();
                const auto w2 = knot2_->getVelocity()->value(), dw2 = knot2_->getAcceleration()->value();
                const Eigen::Matrix<double, 6, 1> xi_21_Jw2 = J_21_inv * w2;
                const Eigen::Matrix<double, 6, 1> omega_12 = -0.5 * slam::liemath::se3::curlyhat(xi_21_Jw2) * w2 + J_21_inv * dw2;

                // Lambda function to compute xi values
                auto compute_xi = [&](int r) {
                    return lambda_.block<6, 6>(r, 6) * w1 + lambda_.block<6, 6>(r, 12) * dw1 +
                        omega_.block<6, 6>(r, 0) * xi_21 + omega_.block<6, 6>(r, 6) * xi_21_Jw2 +
                        omega_.block<6, 6>(r, 12) * omega_12;
                };

                // Compute interpolated values
                const Eigen::Matrix<double, 6, 6> J_i1 = slam::liemath::se3::vec2jac(compute_xi(0));
                const auto w_i = J_i1 * compute_xi(6);
                return J_i1 * (compute_xi(12) + 0.5 * slam::liemath::se3::curlyhat(compute_xi(6)) * w_i);
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto AccelerationInterpolator::forward() const -> slam::eval::Node<OutType>::Ptr {
                // Retrieve state values
                const auto T1 = knot1_->getPose()->forward(), T2 = knot2_->getPose()->forward();
                const auto w1 = knot1_->getVelocity()->forward(), dw1 = knot1_->getAcceleration()->forward();
                const auto w2 = knot2_->getVelocity()->forward(), dw2 = knot2_->getAcceleration()->forward();

                // Compute se(3) algebra of relative transformation and its Jacobian
                const auto xi_21 = (T2->value() / T1->value()).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = slam::liemath::se3::vec2jacinv(xi_21);

                // Precompute reused terms
                const auto w1_val = w1->value(), dw1_val = dw1->value();
                const auto w2_val = w2->value(), dw2_val = dw2->value();
                const Eigen::Matrix<double, 6, 1> xi_21_Jw2 = J_21_inv * w2_val;
                const Eigen::Matrix<double, 6, 1> omega_12 = -0.5 * slam::liemath::se3::curlyhat(xi_21_Jw2) * w2_val + J_21_inv * dw2_val;

                // Compute xi values using lambda function
                auto compute_xi = [&](int r) {
                    return lambda_.block<6, 6>(r, 6) * w1_val + lambda_.block<6, 6>(r, 12) * dw1_val +
                        omega_.block<6, 6>(r, 0) * xi_21 + omega_.block<6, 6>(r, 6) * xi_21_Jw2 +
                        omega_.block<6, 6>(r, 12) * omega_12;
                };

                // Compute interpolated values
                const Eigen::Matrix<double, 6, 6> J_i1 = slam::liemath::se3::vec2jac(compute_xi(0));
                const auto w_i = J_i1 * compute_xi(6);
                OutType dw_i = J_i1 * (compute_xi(12) + 0.5 * slam::liemath::se3::curlyhat(compute_xi(6)) * w_i);

                // Create node and add children efficiently
                const auto node = slam::eval::Node<OutType>::MakeShared(dw_i);
                
                // Explicitly define an iterable type
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
                const auto T1 = knot1_->getPose()->value(), T2 = knot2_->getPose()->value();
                const auto w1 = knot1_->getVelocity()->value(), dw1 = knot1_->getAcceleration()->value();
                const auto w2 = knot2_->getVelocity()->value(), dw2 = knot2_->getAcceleration()->value();

                // Compute se(3) algebra of relative transformation and its Jacobian
                const auto xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = slam::liemath::se3::vec2jacinv(xi_21);

                // Compute xi values using lambda_
                Eigen::Matrix<double, 18, 1> combined;
                combined << w1, dw1, xi_21, J_21_inv * w2, J_21_inv * dw2;
                auto compute_xi = [&](int r) { return lambda_.block<6, 18>(r, 0) * combined; };

                const auto xi_i1 = compute_xi(0), xi_j1 = compute_xi(6), xi_k1 = compute_xi(12);

                // Compute transformation and Jacobians
                const slam::liemath::se3::Transformation T_21_obj(xi_21, 0);
                const Eigen::Matrix<double, 6, 6> J_i1 = slam::liemath::se3::vec2jac(xi_i1);
                const auto w_i = J_i1 * xi_j1;

                // Precompute common Jacobian expressions
                const auto J_prep_2 = J_i1 * (-0.5 * slam::liemath::se3::curlyhat(w_i) +
                                            0.5 * slam::liemath::se3::curlyhat(xi_j1) * J_i1);
                const auto J_prep_3 = -0.25 * J_i1 * slam::liemath::se3::curlyhat(xi_j1) * slam::liemath::se3::curlyhat(xi_j1) -
                                    0.5 * slam::liemath::se3::curlyhat(xi_k1 + 0.5 * slam::liemath::se3::curlyhat(xi_j1) * w_i);

                // Compute partial Jacobian matrix
                const auto w = J_i1 * (omega_.block<6, 18>(12, 0) + 
                                    omega_.block<6, 18>(12, 6) * 0.5 * slam::liemath::se3::curlyhat(w2) + 
                                    omega_.block<6, 18>(12, 12) * 0.5 * slam::liemath::se3::curlyhat(dw2)) * J_21_inv;

                // Process Jacobians efficiently using a lambda-based approach
                std::array<std::function<void()>, 6> jacobian_updates = {
                    [&] { if (knot1_->getPose()->active()) 
                            knot1_->getPose()->backward(lhs * (-w * T_21_obj.adjoint()), 
                                std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(0)), jacs); },
                    [&] { if (knot2_->getPose()->active()) 
                            knot2_->getPose()->backward(lhs * w, 
                                std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(3)), jacs); },
                    [&] { if (knot1_->getVelocity()->active()) 
                            knot1_->getVelocity()->backward(lhs * (J_i1 * lambda_.block<6, 6>(12, 6) +
                                                                J_prep_2 * lambda_.block<6, 6>(6, 6) +
                                                                J_prep_3 * lambda_.block<6, 6>(0, 6)), 
                                std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(1)), jacs); },
                    [&] { if (knot2_->getVelocity()->active()) 
                            knot2_->getVelocity()->backward(lhs * (J_i1 * omega_.block<6, 6>(12, 6) * J_21_inv +
                                                                J_prep_2 * omega_.block<6, 6>(6, 6) * J_21_inv +
                                                                J_prep_3 * omega_.block<6, 6>(0, 6) * J_21_inv), 
                                std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(4)), jacs); },
                    [&] { if (knot2_->getAcceleration()->active()) 
                            knot2_->getAcceleration()->backward(lhs * (J_i1 * omega_.block<6, 6>(12, 12) * J_21_inv +
                                                                    J_prep_2 * omega_.block<6, 6>(6, 12) * J_21_inv +
                                                                    J_prep_3 * omega_.block<6, 6>(0, 12) * J_21_inv), 
                                std::static_pointer_cast<slam::eval::Node<InAccType>>(node->getChild(5)), jacs); }
                };

                for (const auto& update : jacobian_updates) update();
            }

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
