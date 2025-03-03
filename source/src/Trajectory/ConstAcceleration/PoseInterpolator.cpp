#include "source/include/Trajectory/ConstAcceleration/PoseInterpolator.hpp"

#include "source/include/Evaluable/se3/Evaluables.hpp"
#include "source/include/Evaluable/vspace/Evaluables.hpp"
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
            // PoseInterpolator 
            // -----------------------------------------------------------------------------

            PoseInterpolator::PoseInterpolator(const Time& time,
                                               const Variable::ConstPtr& knot1,
                                               const Variable::ConstPtr& knot2)
                : knot1_(knot1), knot2_(knot2) {
                // Calculate time constants
                const double T = (knot2->getTime() - knot1->getTime()).seconds();
                const double tau = (time - knot1->getTime()).seconds();
                const double kappa = (knot2->getTime() - time).seconds();

                // Q and Transition matrix
                const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();

                // Precompute transition and covariance matrices
                const auto Qinv_T = getQinv(T, ones);
                const auto Tran_T = getTran(T);

                // Compute interpolation values
                omega_ = getQ(tau, ones) * getTran(kappa).transpose() * Qinv_T;
                lambda_ = getTran(tau) - omega_ * Tran_T;

            }

            // -----------------------------------------------------------------------------
            // Active
            // -----------------------------------------------------------------------------

            bool PoseInterpolator::active() const {
                return knot1_->getPose()->active() || knot1_->getVelocity()->active() ||
                       knot1_->getAcceleration()->active() || knot2_->getPose()->active() ||
                       knot2_->getVelocity()->active() || knot2_->getAcceleration()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void PoseInterpolator::getRelatedVarKeys(KeySet& keys) const {
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

            auto PoseInterpolator::value() const -> OutType {
                // Retrieve state values from knots
                const auto T1 = knot1_->getPose()->value();
                const auto w1 = knot1_->getVelocity()->value();
                const auto dw1 = knot1_->getAcceleration()->value();
                const auto T2 = knot2_->getPose()->value();
                const auto w2 = knot2_->getVelocity()->value();
                const auto dw2 = knot2_->getAcceleration()->value();

                // Compute the relative transformation in se(3) Lie algebra
                const auto xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Efficiently compute interpolated values
                Eigen::Matrix<double, 18, 1> combined;
                combined << w1, dw1, xi_21, J_21_inv * w2, J_21_inv * dw2;
                
                Eigen::Matrix<double, 6, 1> xi_i1 = lambda_.block<6, 18>(0, 0) * combined;
                Eigen::Matrix<double, 6, 1> xi_j1 = lambda_.block<6, 18>(6, 0) * combined;

                // Compute interpolated transformation matrix using Lie group exponential map
                const liemath::se3::Transformation T_i1(xi_i1, 0);

                // Compute final interpolated pose T_i0
                return T_i1 * T1;
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto PoseInterpolator::forward() const -> slam::eval::Node<OutType>::Ptr {
                // Retrieve state values
                const auto T1 = knot1_->getPose()->forward(), T2 = knot2_->getPose()->forward();
                const auto w1 = knot1_->getVelocity()->forward(), dw1 = knot1_->getAcceleration()->forward();
                const auto w2 = knot2_->getVelocity()->forward(), dw2 = knot2_->getAcceleration()->forward();

                // Compute se(3) transformation and Jacobian
                const auto xi_21 = (T2->value() / T1->value()).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Compute interpolated xi_i1
                const auto Jw2 = J_21_inv * w2->value();
                const auto omega_12 = -0.5 * liemath::se3::curlyhat(Jw2) * w2->value() + J_21_inv * dw2->value();
                const Eigen::Matrix<double, 6, 1> xi_i1 = 
                    lambda_.block<6, 6>(0, 6) * w1->value() + 
                    lambda_.block<6, 6>(0, 12) * dw1->value() + 
                    omega_.block<6, 6>(0, 0) * xi_21 + 
                    omega_.block<6, 6>(0, 6) * Jw2 + 
                    omega_.block<6, 6>(0, 12) * omega_12;

                // Compute interpolated transformation matrix
                const OutType T_i0 = liemath::se3::Transformation(xi_i1,0) * T1->value();
                const auto node = slam::eval::Node<OutType>::MakeShared(T_i0);

                // Explicitly specify the container type
                std::initializer_list<slam::eval::NodeBase::Ptr> children = {T1, w1, dw1, T2, w2, dw2};
                for (const auto& child : children) node->addChild(child);
                
                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void PoseInterpolator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs, 
                                            const slam::eval::Node<OutType>::Ptr& node, 
                                            slam::eval::StateKeyJacobians& jacs) const {
                if (!active()) return;

                // Retrieve state values
                const auto T1 = knot1_->getPose()->value(), T2 = knot2_->getPose()->value();
                const auto w1 = knot1_->getVelocity()->value(), dw1 = knot1_->getAcceleration()->value();
                const auto w2 = knot2_->getVelocity()->value(), dw2 = knot2_->getAcceleration()->value();

                // Compute se(3) transformation and Jacobian
                const auto xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Compute interpolated xi_i1
                const auto Jw2 = J_21_inv * w2;
                const auto omega_12 = -0.5 * liemath::se3::curlyhat(Jw2) * w2 + J_21_inv * dw2;
                const Eigen::Matrix<double, 6, 1> xi_i1 = 
                    lambda_.block<6, 6>(0, 6) * w1 + 
                    lambda_.block<6, 6>(0, 12) * dw1 + 
                    omega_.block<6, 6>(0, 0) * xi_21 + 
                    omega_.block<6, 6>(0, 6) * Jw2 + 
                    omega_.block<6, 6>(0, 12) * omega_12;

                // Compute interpolated transformation and Jacobians
                const liemath::se3::Transformation T_21_obj(xi_21,0);
                const Eigen::Matrix<double, 6, 6> J_i1 = liemath::se3::vec2jac(xi_i1);

                // Precompute common Jacobian expressions
                const auto J_prep = J_i1 * (omega_.block<6, 6>(0, 6) * 0.5 * liemath::se3::curlyhat(w2) +
                                            omega_.block<6, 6>(0, 12) * 0.25 * liemath::se3::curlyhat(w2) * liemath::se3::curlyhat(w2) +
                                            omega_.block<6, 6>(0, 12) * 0.5 * liemath::se3::curlyhat(dw2)) * J_21_inv;

                // Process Jacobians efficiently using a lambda-based approach
                std::array<std::function<void()>, 6> jacobian_updates = {
                    [&] { if (knot1_->getPose()->active()) 
                            knot1_->getPose()->backward(lhs * (-J_prep * T_21_obj.adjoint() + J_i1.adjoint()), 
                                std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(0)), jacs); },
                    [&] { if (knot2_->getPose()->active()) 
                            knot2_->getPose()->backward(lhs * J_prep, 
                                std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(3)), jacs); },
                    [&] { if (knot1_->getVelocity()->active()) 
                            knot1_->getVelocity()->backward(lhs * lambda_.block<6, 6>(0, 6) * J_i1, 
                                std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(1)), jacs); },
                    [&] { if (knot2_->getVelocity()->active()) 
                            knot2_->getVelocity()->backward(lhs * (omega_.block<6, 6>(0, 6) * J_i1 * J_21_inv +
                                                                omega_.block<6, 6>(0, 12) * -0.5 * J_i1 *
                                                                (liemath::se3::curlyhat(Jw2) - liemath::se3::curlyhat(w2) * J_21_inv)), 
                                std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(4)), jacs); },
                    [&] { if (knot1_->getAcceleration()->active()) 
                            knot1_->getAcceleration()->backward(lhs * lambda_.block<6, 6>(0, 12) * J_i1, 
                                std::static_pointer_cast<slam::eval::Node<InAccType>>(node->getChild(2)), jacs); },
                    [&] { if (knot2_->getAcceleration()->active()) 
                            knot2_->getAcceleration()->backward(lhs * omega_.block<6, 6>(0, 12) * J_i1 * J_21_inv, 
                                std::static_pointer_cast<slam::eval::Node<InAccType>>(node->getChild(5)), jacs); }
                };

                for (const auto& update : jacobian_updates) update();
            }
        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
