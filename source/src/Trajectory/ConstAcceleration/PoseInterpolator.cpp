
#include "source/include/Evaluable/se3/Evaluables.hpp"
#include "source/include/Evaluable/vspace/Evaluables.hpp"
#include "source/include/Trajectory/ConstAcceleration/PoseInterpolator.hpp"
#include "source/src/Trajectory/ConstVelocity/Evaluable/JinvVelocityEvaluator.cpp"

#include "source/include/Trajectory/ConstAcceleration/Helper.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------
            auto PoseInterpolator::MakeShared(const Time& time,
                                              const Variable::ConstPtr& knot1,
                                              const Variable::ConstPtr& knot2) -> Ptr {
                return std::make_shared<PoseInterpolator>(time, knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // PoseInterpolator Constructor
            // -----------------------------------------------------------------------------
            PoseInterpolator::PoseInterpolator(const Time& time,
                                               const Variable::ConstPtr& knot1,
                                               const Variable::ConstPtr& knot2)
                : knot1_(knot1), knot2_(knot2) {
                // Calculate time constants
                const double T = (knot2->time() - knot1->time()).seconds();
                const double tau = (time - knot1->time()).seconds();
                const double kappa = (knot2->time() - time).seconds();

                // Q and Transition matrix
                const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
                const auto Q_tau = getQ(tau, ones);
                const auto Qinv_T = getQinv(T, ones);
                const auto Tran_kappa = getTran(kappa);
                const auto Tran_tau = getTran(tau);
                const auto Tran_T = getTran(T);

                // Calculate interpolation values
                omega_ = (Q_tau * Tran_kappa.transpose() * Qinv_T);
                lambda_ = (Tran_tau - omega_ * Tran_T);
            }

            // -----------------------------------------------------------------------------
            // Active
            // -----------------------------------------------------------------------------
            bool PoseInterpolator::active() const {
                return knot1_->pose()->active() || knot1_->velocity()->active() ||
                       knot1_->acceleration()->active() || knot2_->pose()->active() ||
                       knot2_->velocity()->active() || knot2_->acceleration()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------
            void PoseInterpolator::getRelatedVarKeys(KeySet& keys) const {
                knot1_->pose()->getRelatedVarKeys(keys);
                knot1_->velocity()->getRelatedVarKeys(keys);
                knot1_->acceleration()->getRelatedVarKeys(keys);
                knot2_->pose()->getRelatedVarKeys(keys);
                knot2_->velocity()->getRelatedVarKeys(keys);
                knot2_->acceleration()->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto PoseInterpolator::value() const -> OutType {
                // Retrieve state values from knots
                const auto T1 = knot1_->pose()->value();
                const auto w1 = knot1_->velocity()->value();
                const auto dw1 = knot1_->acceleration()->value();
                const auto T2 = knot2_->pose()->value();
                const auto w2 = knot2_->velocity()->value();
                const auto dw2 = knot2_->acceleration()->value();

                // Compute the relative transformation in se(3) Lie algebra
                const auto xi_21 = (T2 / T1).vec();

                // Compute inverse Jacobian of se(3) transformation
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Compute interpolated se(3) Lie algebra element xi_i1
                const Eigen::Matrix<double, 6, 1> xi_i1 =
                    lambda_.block<6, 6>(0, 6) * w1 +
                    lambda_.block<6, 6>(0, 12) * dw1 +
                    omega_.block<6, 6>(0, 0) * xi_21 +
                    omega_.block<6, 6>(0, 6) * J_21_inv * w2 +
                    omega_.block<6, 6>(0, 12) *
                        (-0.5 * liemath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

                // Compute interpolated velocity xi_j1
                const Eigen::Matrix<double, 6, 1> xi_j1 =
                    lambda_.block<6, 6>(6, 6) * w1 +
                    lambda_.block<6, 6>(6, 12) * dw1 +
                    omega_.block<6, 6>(6, 0) * xi_21 +
                    omega_.block<6, 6>(6, 6) * J_21_inv * w2 +
                    omega_.block<6, 6>(6, 12) *
                        (-0.5 * liemath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

                // Compute interpolated transformation matrix using Lie group exponential map
                const liemath::se3::Transformation T_i1(xi_i1,0);

                // Compute final interpolated pose T_i0
                OutType T_i0 = T_i1 * T1;

                return T_i0;
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------
            auto PoseInterpolator::forward() const -> slam::eval::Node<OutType>::Ptr {
                // Forward propagate child nodes
                const auto T1 = knot1_->pose()->forward();
                const auto w1 = knot1_->velocity()->forward();
                const auto dw1 = knot1_->acceleration()->forward();
                const auto T2 = knot2_->pose()->forward();
                const auto w2 = knot2_->velocity()->forward();
                const auto dw2 = knot2_->acceleration()->forward();

                // Compute the relative transformation in se(3) Lie algebra
                const auto xi_21 = (T2->value() / T1->value()).vec();

                // Compute inverse Jacobian of se(3) transformation
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Compute interpolated se(3) Lie algebra element xi_i1
                const Eigen::Matrix<double, 6, 1> xi_i1 =
                    lambda_.block<6, 6>(0, 6) * w1->value() +
                    lambda_.block<6, 6>(0, 12) * dw1->value() +
                    omega_.block<6, 6>(0, 0) * xi_21 +
                    omega_.block<6, 6>(0, 6) * J_21_inv * w2->value() +
                    omega_.block<6, 6>(0, 12) *
                        (-0.5 * liemath::se3::curlyhat(J_21_inv * w2->value()) * w2->value() +
                        J_21_inv * dw2->value());

                // Compute interpolated relative transformation matrix
                const liemath::se3::Transformation T_i1(xi_i1, 0);

                // Compute final interpolated pose T_i0
                OutType T_i0 = T_i1 * T1->value();

                // Create new node for interpolated transformation
                const auto node = slam::eval::Node<OutType>::MakeShared(T_i0);
                node->addChild(T1);
                node->addChild(w1);
                node->addChild(dw1);
                node->addChild(T2);
                node->addChild(w2);
                node->addChild(dw2);

                // Corrected active() calls
                if (knot1_->pose()->active()) node->addChild(T1);
                if (knot1_->velocity()->active()) node->addChild(w1);
                if (knot1_->acceleration()->active()) node->addChild(dw1);
                if (knot2_->pose()->active()) node->addChild(T2);
                if (knot2_->velocity()->active()) node->addChild(w2);
                if (knot2_->acceleration()->active()) node->addChild(dw2);

                return node;
            }
            
            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            void PoseInterpolator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                              const eval::Node<OutType>::Ptr& node,
                              eval::StateKeyJacobians& jacs) const {
                if (!active()) return;

                // Retrieve values from knots
                const auto T1 = knot1_->pose()->value();
                const auto w1 = knot1_->velocity()->value();
                const auto dw1 = knot1_->acceleration()->value();
                const auto T2 = knot2_->pose()->value();
                const auto w2 = knot2_->velocity()->value();
                const auto dw2 = knot2_->acceleration()->value();

                // Compute relative transformation in se(3) Lie algebra
                const auto xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Compute interpolated se(3) Lie algebra element xi_i1
                const Eigen::Matrix<double, 6, 1> xi_i1 =
                    lambda_.block<6, 6>(0, 6) * w1 +
                    lambda_.block<6, 6>(0, 12) * dw1 +
                    omega_.block<6, 6>(0, 0) * xi_21 +
                    omega_.block<6, 6>(0, 6) * J_21_inv * w2 +
                    omega_.block<6, 6>(0, 12) *
                        (-0.5 * liemath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

                // Compute transformation matrices
                const liemath::se3::Transformation T_21(xi_21,0);
                const liemath::se3::Transformation T_i1(xi_i1,0);

                // Compute Jacobian associated with the interpolated relative transformation
                const Eigen::Matrix<double, 6, 6> J_i1 = liemath::se3::vec2jac(xi_i1);

                // -------------------------------
                // 🔹 Compute Jacobians for Pose
                // -------------------------------
                if (knot1_->pose()->active() || knot2_->pose()->active()) {
                    // Precompute transformation-dependent Jacobians
                    const Eigen::Matrix<double, 6, 6> J_rel =
                        J_i1 *
                        (omega_.block<6, 6>(0, 0) * Eigen::Matrix<double, 6, 6>::Identity() +
                        omega_.block<6, 6>(0, 6) * 0.5 * liemath::se3::curlyhat(w2) +
                        omega_.block<6, 6>(0, 12) * 0.25 * liemath::se3::curlyhat(w2) *
                            liemath::se3::curlyhat(w2) +
                        omega_.block<6, 6>(0, 12) * 0.5 * liemath::se3::curlyhat(dw2)) *
                        J_21_inv;

                    if (knot1_->pose()->active()) {
                        const auto T1_ = std::dynamic_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(0));
                        Eigen::MatrixXd new_lhs = lhs * (-J_rel * T_21.adjoint() + T_i1.adjoint());
                        knot1_->pose()->backward(new_lhs, T1_, jacs);
                    }

                    if (knot2_->pose()->active()) {
                        const auto T2_ = std::dynamic_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(3));
                        Eigen::MatrixXd new_lhs = lhs * J_rel;
                        knot2_->pose()->backward(new_lhs, T2_, jacs);
                    }
                }

                // -------------------------------
                // 🔹 Compute Jacobians for Velocity
                // -------------------------------
                if (knot1_->velocity()->active()) {
                    const auto w1_ = std::dynamic_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(1));
                    Eigen::MatrixXd new_lhs = lhs * lambda_.block<6, 6>(0, 6) * J_i1;
                    knot1_->velocity()->backward(new_lhs, w1_, jacs);
                }

                if (knot2_->velocity()->active()) {
                    const auto w2_ = std::dynamic_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(4));
                    Eigen::MatrixXd new_lhs =
                        lhs * (omega_.block<6, 6>(0, 6) * J_i1 * J_21_inv +
                            omega_.block<6, 6>(0, 12) * -0.5 * J_i1 *
                                (liemath::se3::curlyhat(J_21_inv * w2) -
                                    liemath::se3::curlyhat(w2) * J_21_inv));
                    knot2_->velocity()->backward(new_lhs, w2_, jacs);
                }

                // -------------------------------
                // 🔹 Compute Jacobians for Acceleration
                // -------------------------------
                if (knot1_->acceleration()->active()) {
                    const auto dw1_ = std::dynamic_pointer_cast<slam::eval::Node<InAccType>>(node->getChild(2));
                    Eigen::MatrixXd new_lhs = lhs * lambda_.block<6, 6>(0, 12) * J_i1;
                    knot1_->acceleration()->backward(new_lhs, dw1_, jacs);
                }

                if (knot2_->acceleration()->active()) {
                    const auto dw2_ = std::dynamic_pointer_cast<slam::eval::Node<InAccType>>(node->getChild(5));
                    Eigen::MatrixXd new_lhs = lhs * omega_.block<6, 6>(0, 12) * J_i1 * J_21_inv;
                    knot2_->acceleration()->backward(new_lhs, dw2_, jacs);
                }
            }
        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
