#include "source/include/Trajectory/ConstAcceleration/VelocityInterpolator.hpp"
#include "source/include/Trajectory/ConstAcceleration/Helper.hpp"

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

                const double T = (knot2->time() - knot1->time()).seconds();
                const double tau = (time - knot1->time()).seconds();
                const double kappa = (knot2->time() - time).seconds();

                const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
                const auto Q_tau = getQ(tau, ones);
                const auto Qinv_T = getQinv(T, ones);
                const auto Tran_kappa = getTran(kappa);
                const auto Tran_tau = getTran(tau);
                const auto Tran_T = getTran(T);

                omega_ = (Q_tau * Tran_kappa.transpose() * Qinv_T);
                lambda_ = (Tran_tau - omega_ * Tran_T);
            }

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------

            bool VelocityInterpolator::active() const {
                return knot1_->pose()->active() || knot1_->velocity()->active() ||
                    knot1_->acceleration()->active() || knot2_->pose()->active() ||
                    knot2_->velocity()->active() || knot2_->acceleration()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void VelocityInterpolator::getRelatedVarKeys(eval::Evaluable<InPoseType>::KeySet& keys) const {
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

            auto VelocityInterpolator::value() const -> OutType {
                // Retrieve pose, velocity, and acceleration values for knots
                const auto T1 = knot1_->pose()->value();
                const auto w1 = knot1_->velocity()->value();
                const auto dw1 = knot1_->acceleration()->value();
                const auto T2 = knot2_->pose()->value();
                const auto w2 = knot2_->velocity()->value();
                const auto dw2 = knot2_->acceleration()->value();

                // Compute SE(3) algebra representation of relative transformation
                const auto xi_21 = (T2 / T1).vec();
                
                // Compute inverse left Jacobian of xi_21
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Compute interpolated relative se(3) algebra terms
                const Eigen::Matrix<double, 6, 1> xi_i1 =
                    lambda_.block<6, 6>(0, 6) * w1 +
                    lambda_.block<6, 6>(0, 12) * dw1 +
                    omega_.block<6, 6>(0, 0) * xi_21 +
                    omega_.block<6, 6>(0, 6) * J_21_inv * w2 +
                    omega_.block<6, 6>(0, 12) * (-0.5 * liemath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

                const Eigen::Matrix<double, 6, 1> xi_j1 =
                    lambda_.block<6, 6>(6, 6) * w1 +
                    lambda_.block<6, 6>(6, 12) * dw1 +
                    omega_.block<6, 6>(6, 0) * xi_21 +
                    omega_.block<6, 6>(6, 6) * J_21_inv * w2 +
                    omega_.block<6, 6>(6, 12) * (-0.5 * liemath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

                // Compute final interpolated velocity in SE(3)
                OutType w_i = liemath::se3::vec2jac(xi_i1) * xi_j1;

                return w_i;
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto VelocityInterpolator::forward() const -> eval::Node<OutType>::Ptr {
                const auto T1 = knot1_->pose()->forward();
                const auto w1 = knot1_->velocity()->forward();
                const auto dw1 = knot1_->acceleration()->forward();
                const auto T2 = knot2_->pose()->forward();
                const auto w2 = knot2_->velocity()->forward();
                const auto dw2 = knot2_->acceleration()->forward();

                // Compute relative transformation algebra
                const auto xi_21 = (T2->value() / T1->value()).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Compute interpolated relative se(3) algebra
                const Eigen::Matrix<double, 6, 1> xi_i1 =
                    lambda_.block<6, 6>(0, 6) * w1->value() +
                    lambda_.block<6, 6>(0, 12) * dw1->value() +
                    omega_.block<6, 6>(0, 0) * xi_21 +
                    omega_.block<6, 6>(0, 6) * J_21_inv * w2->value() +
                    omega_.block<6, 6>(0, 12) * (-0.5 * liemath::se3::curlyhat(J_21_inv * w2->value()) * w2->value() + J_21_inv * dw2->value());

                const Eigen::Matrix<double, 6, 1> xi_j1 =
                    lambda_.block<6, 6>(6, 6) * w1->value() +
                    lambda_.block<6, 6>(6, 12) * dw1->value() +
                    omega_.block<6, 6>(6, 0) * xi_21 +
                    omega_.block<6, 6>(6, 6) * J_21_inv * w2->value() +
                    omega_.block<6, 6>(6, 12) * (-0.5 * liemath::se3::curlyhat(J_21_inv * w2->value()) * w2->value() + J_21_inv * dw2->value());

                // Compute interpolated velocity
                OutType w_i = liemath::se3::vec2jac(xi_i1) * xi_j1;

                // Create computation node
                const auto node = eval::Node<OutType>::MakeShared(w_i);
                node->addChild(T1);
                node->addChild(w1);
                node->addChild(dw1);
                node->addChild(T2);
                node->addChild(w2);
                node->addChild(dw2);

                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void VelocityInterpolator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                                const eval::Node<OutType>::Ptr& node,
                                                eval::StateKeyJacobians& jacs) const {
                if (!active()) return;

                // Retrieve values from trajectory knots
                const auto T1 = knot1_->pose()->value();
                const auto w1 = knot1_->velocity()->value();
                const auto dw1 = knot1_->acceleration()->value();
                const auto T2 = knot2_->pose()->value();
                const auto w2 = knot2_->velocity()->value();
                const auto dw2 = knot2_->acceleration()->value();

                // Compute relative transformation algebra
                const auto xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Compute interpolated se(3) algebra
                const Eigen::Matrix<double, 6, 1> xi_i1 =
                    lambda_.block<6, 6>(0, 6) * w1 +
                    lambda_.block<6, 6>(0, 12) * dw1 +
                    omega_.block<6, 6>(0, 0) * xi_21 +
                    omega_.block<6, 6>(0, 6) * J_21_inv * w2 +
                    omega_.block<6, 6>(0, 12) * (-0.5 * liemath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

                const Eigen::Matrix<double, 6, 1> xi_j1 =
                    lambda_.block<6, 6>(6, 6) * w1 +
                    lambda_.block<6, 6>(6, 12) * dw1 +
                    omega_.block<6, 6>(6, 0) * xi_21 +
                    omega_.block<6, 6>(6, 6) * J_21_inv * w2 +
                    omega_.block<6, 6>(6, 12) * (-0.5 * liemath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

                // Compute transformation matrix
                const liemath::se3::Transformation T_21(xi_21,0);

                // Compute Jacobians
                const Eigen::Matrix<double, 6, 6> J_i1 = liemath::se3::vec2jac(xi_i1);
                const Eigen::Matrix<double, 6, 6> xi_j1_ch = -0.5 * liemath::se3::curlyhat(xi_j1);

                if (knot1_->pose()->active() || knot2_->pose()->active()) {
                    const Eigen::Matrix<double, 6, 6> w =
                        J_i1 * (omega_.block<6, 6>(6, 0) * Eigen::Matrix<double, 6, 6>::Identity() +
                                omega_.block<6, 6>(6, 6) * 0.5 * liemath::se3::curlyhat(w2) +
                                omega_.block<6, 6>(6, 12) * 0.25 * liemath::se3::curlyhat(w2) * liemath::se3::curlyhat(w2) +
                                omega_.block<6, 6>(6, 12) * 0.5 * liemath::se3::curlyhat(dw2)) * J_21_inv +
                        xi_j1_ch * (omega_.block<6, 6>(0, 0) * Eigen::Matrix<double, 6, 6>::Identity() +
                                    omega_.block<6, 6>(0, 6) * 0.5 * liemath::se3::curlyhat(w2) +
                                    omega_.block<6, 6>(0, 12) * 0.25 * liemath::se3::curlyhat(w2) * liemath::se3::curlyhat(w2) +
                                    omega_.block<6, 6>(0, 12) * 0.5 * liemath::se3::curlyhat(dw2)) * J_21_inv;

                    if (knot1_->pose()->active()) {
                        const auto T1_ = std::static_pointer_cast<eval::Node<InPoseType>>(node->getChild(0));
                        knot1_->pose()->backward(lhs * (-w * T_21.adjoint()), T1_, jacs);
                    }
                    if (knot2_->pose()->active()) {
                        const auto T2_ = std::static_pointer_cast<eval::Node<InPoseType>>(node->getChild(3));
                        knot2_->pose()->backward(lhs * w, T2_, jacs);
                    }
                }
            }

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
