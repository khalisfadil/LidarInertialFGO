#include "source/include/Trajectory/ConstVelocity/PoseExtrapolator.hpp"

#include "source/include/Evaluable/se3/Evaluables.hpp"
#include "source/include/Evaluable/vspace/Evaluables.hpp"
#include "source/include/Trajectory/ConstVelocity/Helper.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            PoseExtrapolator::Ptr PoseExtrapolator::MakeShared(
                const slam::traj::Time& time, const Variable::ConstPtr& knot) {
                return std::make_shared<PoseExtrapolator>(time, knot);
            }

            // -----------------------------------------------------------------------------
            // PoseExtrapolator
            // -----------------------------------------------------------------------------
            
            PoseExtrapolator::PoseExtrapolator(
                const slam::traj::Time& time, const Variable::ConstPtr& knot)
                : knot_(knot) {

                // Compute transition matrix Phi based on time difference
                const double tau = (time - knot->getTime()).seconds();
                Phi_ = getTran(tau);
            }

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------
            
            bool PoseExtrapolator::active() const {
                return knot_->getPose()->active() || knot_->getVelocity()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------
            
            void PoseExtrapolator::getRelatedVarKeys(KeySet& keys) const {
                knot_->getPose()->getRelatedVarKeys(keys);
                knot_->getVelocity()->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------
            
            auto PoseExtrapolator::value() const -> OutType {
                const auto& T_k = knot_->getPose()->value();
                const auto& w_k = knot_->getVelocity()->value();

                // Compute relative transformation from velocity
                const Eigen::Matrix<double, 6, 1> xi_k =
                    Phi_.block<6, 6>(0, 6) * w_k;

                // Convert xi_k to transformation matrix and apply it
                return slam::liemath::se3::Transformation(xi_k, 0) * T_k;
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------
            
            auto PoseExtrapolator::forward() const -> slam::eval::Node<OutType>::Ptr {
                const auto T_k = knot_->getPose()->forward();
                const auto w_k = knot_->getVelocity()->forward();

                const auto interpolated_value = this->value();
                auto node = slam::eval::Node<OutType>::MakeShared(interpolated_value);

                node->addChild(T_k);
                node->addChild(w_k);

                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------
            
            void PoseExtrapolator::backward(
                const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                const slam::eval::Node<OutType>::Ptr& node,
                slam::eval::StateKeyJacobians& jacs) const {

                if (!active()) return;

                const auto& w_k = knot_->getVelocity()->value();
                const Eigen::Matrix<double, 6, 1> xi_k =
                    Phi_.block<6, 6>(0, 6) * w_k;

                // Compute Jacobian matrix
                const Eigen::Matrix<double, 6, 6> J_k = slam::liemath::se3::vec2jac(xi_k);
                const slam::liemath::se3::Transformation T_k(xi_k,0);

                if (knot_->getPose()->active()) {
                    const auto T_k_ = std::dynamic_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(0));
                    knot_->getPose()->backward(lhs * T_k.adjoint(), T_k_, jacs);
                }
                if (knot_->getVelocity()->active()) {
                    const auto w_k_ = std::dynamic_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(1));
                    knot_->getVelocity()->backward(lhs * J_k * Phi_.block<6, 6>(0, 6), w_k_, jacs);
                }
            }

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
