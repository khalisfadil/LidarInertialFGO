#include "Trajectory/ConstVelocity/PoseInterpolator.hpp"

#include "Evaluable/se3/Evaluables.hpp"
#include "Evaluable/vspace/Evaluables.hpp"
#include "Trajectory/ConstVelocity/Helper.hpp"
#include "Trajectory/ConstVelocity/Evaluable/JinvVelocityEvaluator.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {
            
            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            PoseInterpolator::Ptr PoseInterpolator::MakeShared(
                const slam::traj::Time& time,
                const Variable::ConstPtr& knot1,
                const Variable::ConstPtr& knot2) {
                return std::make_shared<PoseInterpolator>(time, knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // PoseInterpolator
            // -----------------------------------------------------------------------------

            PoseInterpolator::PoseInterpolator(const slam::traj::Time& time,
                                                const Variable::ConstPtr& knot1,
                                                const Variable::ConstPtr& knot2)
                : knot1_(knot1), knot2_(knot2) {
                // Calculate time constants
                const double T = (knot2->getTime() - knot1->getTime()).seconds();
                const double tau = (time - knot1->getTime()).seconds();
                const double ratio = tau / T;
                const double ratio2 = ratio * ratio;

                // Precompute common terms
                const double r2_minus_r = ratio2 - ratio;
                const double r_minus_r2 = ratio - ratio2;

                // Calculate 'psi' interpolation values
                psi11_ = ratio2 * (3.0 - 2.0 * ratio);  // 3 * ratio^2 - 2 * ratio^3
                psi12_ = tau * r2_minus_r;              // tau * (ratio^2 - ratio)
                psi21_ = 6.0 * r_minus_r2 / T;          // 6 * (ratio - ratio^2) / T
                psi22_ = ratio * (3.0 * ratio - 2.0);   // 3 * ratio^2 - 2 * ratio

                // Calculate 'lambda' interpolation values
                lambda11_ = 1.0 - psi11_;
                lambda12_ = tau * (1.0 - r2_minus_r - 2.0 * r_minus_r2);  // tau - T * psi11 - psi12
                lambda21_ = -psi21_;
                lambda22_ = 1.0 + (T * -6.0 - 3.0 * tau) * r_minus_r2;    // 1 - T * psi21 - psi22
            }

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------

            bool PoseInterpolator::active() const {
                return knot1_->getPose()->active() || knot1_->getVelocity()->active() ||
                       knot2_->getPose()->active() || knot2_->getVelocity()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void PoseInterpolator::getRelatedVarKeys(KeySet& keys) const {
                knot1_->getPose()->getRelatedVarKeys(keys);
                knot1_->getVelocity()->getRelatedVarKeys(keys);
                knot2_->getPose()->getRelatedVarKeys(keys);
                knot2_->getVelocity()->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto PoseInterpolator::value() const -> OutType {
                // Retrieve state values
                const auto T1 = knot1_->getPose()->value();
                const auto T2 = knot2_->getPose()->value();
                const auto w1 = knot1_->getVelocity()->value();
                const auto w2 = knot2_->getVelocity()->value();

                // Compute SE(3) algebra and Jacobian
                const Eigen::Matrix<double, 6, 1> xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Precompute reused term
                const Eigen::Matrix<double, 6, 1> J_w2 = J_21_inv * w2;

                // Calculate interpolated relative SE(3) algebra
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda12_ * w1 + psi11_ * xi_21 + psi12_ * J_w2;

                // Return interpolated transformation
                return slam::liemath::se3::Transformation(xi_i1,0) * T1;
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto PoseInterpolator::forward() const -> slam::eval::Node<OutType>::Ptr {
                // Retrieve state values
                const auto T1 = knot1_->getPose()->forward();
                const auto w1 = knot1_->getVelocity()->forward();
                const auto T2 = knot2_->getPose()->forward();
                const auto w2 = knot2_->getVelocity()->forward();

                // Precompute values and Jacobian
                const auto T1_val = T1->value();
                const Eigen::Matrix<double, 6, 1> xi_21 = (T2->value() / T1_val).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);
                const Eigen::Matrix<double, 6, 1> J_w2 = J_21_inv * w2->value();

                // Calculate interpolated relative SE(3) algebra
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda12_ * w1->value() + psi11_ * xi_21 + psi12_ * J_w2;

                // Create node with interpolated transformation
                const auto node = slam::eval::Node<OutType>::MakeShared(slam::liemath::se3::Transformation(xi_i1,0) * T1_val);
                node->addChild(T1);
                node->addChild(w1);
                node->addChild(T2);
                node->addChild(w2);

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
                const auto T1 = knot1_->getPose()->value();
                const auto T2 = knot2_->getPose()->value();
                const auto w1 = knot1_->getVelocity()->value();
                const auto w2 = knot2_->getVelocity()->value();

                // Compute SE(3) algebra and Jacobians
                const Eigen::Matrix<double, 6, 1> xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);
                const Eigen::Matrix<double, 6, 1> J_w2 = J_21_inv * w2;
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda12_ * w1 + psi11_ * xi_21 + psi12_ * J_w2;
                const Eigen::Matrix<double, 6, 6> J_i1 = liemath::se3::vec2jac(xi_i1);

                // Precompute transformation and adjoint
                const liemath::se3::Transformation T_21(xi_21,0);
                const liemath::se3::Transformation T_i1(xi_i1,0);
                const Eigen::Matrix<double, 6, 6> T_21_adj = T_21.adjoint();
                const Eigen::Matrix<double, 6, 6> T_i1_adj = T_i1.adjoint();

                // Precompute common Jacobian term
                const Eigen::Matrix<double, 6, 6> w = J_i1 * (psi11_ * J_21_inv + psi12_ * 0.5 * liemath::se3::curlyhat(w2) * J_21_inv);

                // Lambda-based Jacobian updates
                std::array<std::function<void()>, 4> updates = {
                    [&]() {
                        if (knot1_->getPose()->active()) {
                            const auto T1_ = std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(0));
                            knot1_->getPose()->backward(lhs * (-w * T_21_adj + T_i1_adj), T1_, jacs);
                        }
                    },
                    [&]() {
                        if (knot2_->getPose()->active()) {
                            const auto T2_ = std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(2));
                            knot2_->getPose()->backward(lhs * w, T2_, jacs);
                        }
                    },
                    [&]() {
                        if (knot1_->getVelocity()->active()) {
                            const auto w1_ = std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(1));
                            knot2_->getVelocity()->backward(lhs * (lambda12_ * J_i1), w1_, jacs);
                        }
                    },
                    [&]() {
                        if (knot2_->getVelocity()->active()) {
                            const auto w2_ = std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(3));
                            knot2_->getVelocity()->backward(lhs * (psi12_ * J_i1 * J_21_inv), w2_, jacs);
                        }
                    }
                };

                // Execute updates
                for (const auto& update : updates) update();
            }
        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
