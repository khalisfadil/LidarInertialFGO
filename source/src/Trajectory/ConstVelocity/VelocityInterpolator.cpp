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

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
