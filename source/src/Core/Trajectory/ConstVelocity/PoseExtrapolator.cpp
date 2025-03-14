#include "Core/Trajectory/ConstVelocity/PoseExtrapolator.hpp"

#include "Core/Evaluable/se3/Evaluables.hpp"
#include "Core/Evaluable/vspace/Evaluables.hpp"
#include "Core/Trajectory/ConstVelocity/Helper.hpp"

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
                // Retrieve state values
                const auto T = knot_->getPose()->forward();
                const auto w = knot_->getVelocity()->forward();

                // Create node with extrapolated transformation
                const auto node = slam::eval::Node<OutType>::MakeShared(
                    slam::liemath::se3::Transformation(Phi_.block<6, 6>(0, 6) * w->value(),0) * T->value()
                );
                node->addChild(T);
                node->addChild(w);

                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void PoseExtrapolator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                                const slam::eval::Node<OutType>::Ptr& node,
                                                slam::eval::StateKeyJacobians& jacs) const {
                if (!active()) return;

                // Retrieve state value and compute SE(3) algebra
                const auto w = knot_->getVelocity()->value();
                const Eigen::Matrix<double, 6, 1> xi_i1 = Phi_.block<6, 6>(0, 6) * w;

                // Precompute Jacobians and adjoint
                const Eigen::Matrix<double, 6, 6> J_i1 = liemath::se3::vec2jac(xi_i1);
                const Eigen::Matrix<double, 6, 6> T_i1_adj = liemath::se3::Transformation(xi_i1,0).adjoint();

                // Lambda-based Jacobian updates
                std::array<std::function<void()>, 2> updates = {
                    [&]() {
                        if (knot_->getPose()->active()) {
                            const auto T_ = std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(0));
                            knot_->getPose()->backward(lhs * T_i1_adj, T_, jacs);
                        }
                    },
                    [&]() {
                        if (knot_->getVelocity()->active()) {
                            const auto w_ = std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(1));
                            knot_->getVelocity()->backward(lhs * J_i1 * Phi_.block<6, 6>(0, 6), w_, jacs);
                        }
                    }
                };

                // Execute updates
                for (const auto& update : updates) update();
            }

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
