#include "source/include/Trajectory/ConstVelocity/PoseInterpolator.hpp"

#include "source/include/Evaluable/se3/Evaluables.hpp"
#include "source/include/Evaluable/vspace/Evaluables.hpp"
#include "source/include/Trajectory/ConstVelocity/Helper.hpp"

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

            PoseInterpolator::PoseInterpolator(
                const slam::traj::Time& time,
                const Variable::ConstPtr& knot1,
                const Variable::ConstPtr& knot2)
                : knot1_(knot1), knot2_(knot2) {

                // Compute time parameters
                const double T = (knot2->getTime() - knot1->getTime()).seconds();
                const double tau = (time - knot1->getTime()).seconds();

                // Compute interpolation coefficients
                const double ratio = tau / T;
                const double ratio2 = ratio * ratio;
                const double ratio3 = ratio2 * ratio;

                psi11_ = 3.0 * ratio2 - 2.0 * ratio3;
                psi12_ = tau * (ratio2 - ratio);
                psi21_ = 6.0 * (ratio - ratio2) / T;
                psi22_ = 3.0 * ratio2 - 2.0 * ratio;

                lambda11_ = 1.0 - psi11_;
                lambda12_ = tau - T * psi11_ - psi12_;
                lambda21_ = -psi21_;
                lambda22_ = 1.0 - T * psi21_ - psi22_;
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
                const auto& T1 = knot1_->getPose()->value();
                const auto& w1 = knot1_->getVelocity()->value();
                const auto& T2 = knot2_->getPose()->value();
                const auto& w2 = knot2_->getVelocity()->value();

                // Compute relative pose
                const auto xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = slam::liemath::se3::vec2jacinv(xi_21);

                // Compute interpolated relative se(3) vector
                const Eigen::Matrix<double, 6, 1> xi_i1 =
                    lambda12_ * w1 + psi11_ * xi_21 + psi12_ * J_21_inv * w2;

                // Convert interpolated se(3) vector to transformation using constructor
                return slam::liemath::se3::Transformation(xi_i1, 0) * T1;
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto PoseInterpolator::forward() const -> slam::eval::Node<OutType>::Ptr {
                const auto T1 = knot1_->getPose()->forward();
                const auto w1 = knot1_->getVelocity()->forward();
                const auto T2 = knot2_->getPose()->forward();
                const auto w2 = knot2_->getVelocity()->forward();

                auto interpolated_value = this->value();
                auto node = slam::eval::Node<OutType>::MakeShared(interpolated_value);

                node->addChild(T1);
                node->addChild(w1);
                node->addChild(T2);
                node->addChild(w2);

                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void PoseInterpolator::backward(
                const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                const slam::eval::Node<OutType>::Ptr& node,
                slam::eval::StateKeyJacobians& jacs) const {

                if (!active()) return;

                const auto& T1 = knot1_->getPose()->value();
                const auto& w1 = knot1_->getVelocity()->value();
                const auto& T2 = knot2_->getPose()->value();
                const auto& w2 = knot2_->getVelocity()->value();

                const auto xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = slam::liemath::se3::vec2jacinv(xi_21);

                if (knot1_->getPose()->active()) {  // FIXED
                    const auto T1_ = std::dynamic_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(0));
                    knot1_->getPose()->backward(lhs * (-psi11_ * J_21_inv), T1_, jacs);
                }
                if (knot2_->getPose()->active()) {  // FIXED
                    const auto T2_ = std::dynamic_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(2));
                    knot2_->getPose()->backward(lhs * psi11_, T2_, jacs);
                }
            }
        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
