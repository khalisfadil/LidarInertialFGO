#include "source/include/Trajectory/ConstVelocity/VelocityInterpolator.hpp"

#include "source/include/Evaluable/se3/Evaluables.hpp"
#include "source/include/Evaluable/vspace/Evaluables.hpp"
#include "source/include/Trajectory/ConstVelocity/Helper.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {
            
            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            VelocityInterpolator::Ptr VelocityInterpolator::MakeShared(
                const slam::traj::Time& time,
                const Variable::ConstPtr& knot1,
                const Variable::ConstPtr& knot2) {
                return std::make_shared<VelocityInterpolator>(time, knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // VelocityInterpolator
            // -----------------------------------------------------------------------------

            VelocityInterpolator::VelocityInterpolator(
                const slam::traj::Time& time,
                const Variable::ConstPtr& knot1,
                const Variable::ConstPtr& knot2)
                : knot1_(knot1), knot2_(knot2) {

                // Compute time parameters
                const double T = (knot2->getTime() - knot1->getTime()).seconds();
                const double tau = (time - knot1->getTime()).seconds();
                const double kappa = (knot2->getTime() - time).seconds();
                
                // Compute interpolation matrices
                const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
                const auto Q_tau = getQ(tau, ones);
                const auto Qinv_T = getQinv(T, ones);
                const auto Tran_kappa = getTran(kappa);
                const auto Tran_tau = getTran(tau);
                const auto Tran_T = getTran(T);

                // Compute psi and lambda interpolation values
                const auto psi = (Q_tau * Tran_kappa.transpose() * Qinv_T);
                const auto lambda = (Tran_tau - psi * Tran_T);

                psi11_ = psi(0, 0);
                psi12_ = psi(0, 6);
                psi21_ = psi(6, 0);
                psi22_ = psi(6, 6);
                lambda11_ = lambda(0, 0);
                lambda12_ = lambda(0, 6);
                lambda21_ = lambda(6, 0);
                lambda22_ = lambda(6, 6);
            }

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------

            bool VelocityInterpolator::active() const {
                return knot1_->getPose()->active() || knot1_->getVelocity()->active() ||
                       knot2_->getPose()->active() || knot2_->getVelocity()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void VelocityInterpolator::getRelatedVarKeys(KeySet& keys) const {
                knot1_->getPose()->getRelatedVarKeys(keys);
                knot1_->getVelocity()->getRelatedVarKeys(keys);
                knot2_->getPose()->getRelatedVarKeys(keys);
                knot2_->getVelocity()->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto VelocityInterpolator::value() const -> OutType {
                const auto T1 = knot1_->getPose()->value();
                const auto w1 = knot1_->getVelocity()->value();
                const auto T2 = knot2_->getPose()->value();
                const auto w2 = knot2_->getVelocity()->value();

                const auto xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = slam::liemath::se3::vec2jacinv(xi_21);

                return slam::liemath::se3::vec2jac(
                    lambda12_ * w1 + psi11_ * xi_21 + psi12_ * J_21_inv * w2
                ) * (
                    lambda22_ * w1 + psi21_ * xi_21 + psi22_ * J_21_inv * w2
                );
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto VelocityInterpolator::forward() const -> slam::eval::Node<OutType>::Ptr {
                const auto T1 = knot1_->getPose()->forward();
                const auto w1 = knot1_->getVelocity()->forward();
                const auto T2 = knot2_->getPose()->forward();
                const auto w2 = knot2_->getVelocity()->forward();

                const auto value = this->value();

                auto node = slam::eval::Node<OutType>::MakeShared(value);
                node->addChild(T1);
                node->addChild(w1);
                node->addChild(T2);
                node->addChild(w2);

                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void VelocityInterpolator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                    const slam::eval::Node<OutType>::Ptr& node,
                                    slam::eval::StateKeyJacobians& jacs) const {
                if (!active()) return;

                // Retrieve state values
                const auto T1 = knot1_->getPose()->value();
                const auto w1 = knot1_->getVelocity()->value();
                const auto T2 = knot2_->getPose()->value();
                const auto w2 = knot2_->getVelocity()->value();

                // Compute se(3) algebra of relative transformation
                const auto xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = slam::liemath::se3::vec2jacinv(xi_21);
                
                // Compute interpolated values efficiently
                Eigen::Matrix<double, 12, 1> combined;
                combined << w1, xi_21, J_21_inv * w2;
                
                Eigen::Matrix<double, 6, 1> xi_i1 = lambda12_ * w1 + psi11_ * xi_21 + psi12_ * J_21_inv * w2;
                Eigen::Matrix<double, 6, 1> xi_j1 = lambda22_ * w1 + psi21_ * xi_21 + psi22_ * J_21_inv * w2;
                
                const Eigen::Matrix<double, 6, 6> J_i1 = slam::liemath::se3::vec2jac(xi_i1);
                const auto w_i = J_i1 * xi_j1;
                const auto J_prep_2 = J_i1 * (-0.5 * slam::liemath::se3::curlyhat(w_i) +
                                                0.5 * slam::liemath::se3::curlyhat(xi_j1) * J_i1);
                const auto J_prep_3 = -0.25 * J_i1 * slam::liemath::se3::curlyhat(xi_j1) *
                                            slam::liemath::se3::curlyhat(xi_j1) -
                                        0.5 * slam::liemath::se3::curlyhat(w_i);

                // Compute relative transformation matrix
                const slam::liemath::se3::Transformation T_21 = T2 * T1.inverse();

                // Process Jacobians efficiently
                std::array<std::pair<int, std::function<void()>>, 4> jacobian_updates = {
                    std::make_pair(0, [&] {
                        if (knot1_->getPose()->active()) {
                            auto T1_ = std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(0));
                            knot1_->getPose()->backward(lhs * (-J_prep_2 * T_21.adjoint()), T1_, jacs);
                        }
                    }),
                    std::make_pair(2, [&] {
                        if (knot2_->getPose()->active()) {
                            auto T2_ = std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(2));
                            knot2_->getPose()->backward(lhs * J_prep_2, T2_, jacs);
                        }
                    }),
                    std::make_pair(1, [&] {
                        if (knot1_->getVelocity()->active()) {
                            auto w1_ = std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(1));
                            knot1_->getVelocity()->backward(lhs * (J_i1 * lambda22_ + J_prep_3 * lambda12_), w1_, jacs);
                        }
                    }),
                    std::make_pair(3, [&] {
                        if (knot2_->getVelocity()->active()) {
                            auto w2_ = std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(3));
                            knot2_->getVelocity()->backward(lhs * (J_i1 * psi22_ * J_21_inv + J_prep_3 * psi12_ * J_21_inv), w2_, jacs);
                        }
                    })};
                
                for (const auto& update : jacobian_updates) {
                    update.second();
                }
                }


        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
