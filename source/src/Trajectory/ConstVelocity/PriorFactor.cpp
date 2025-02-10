#include "source/include/Trajectory/ConstVelocity/PriorFactor.hpp"

#include "source/include/Evaluable/se3/Evaluables.hpp"
#include "source/include/Evaluable/vspace/Evaluables.hpp"
#include "source/include/Trajectory/ConstVelocity/Helper.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {
            
            // -----------------------------------------------------------------------------
            // PriorFactor
            // -----------------------------------------------------------------------------

            PriorFactor::PriorFactor(const Variable::ConstPtr& knot1,
                                     const Variable::ConstPtr& knot2)
                : knot1_(knot1), knot2_(knot2) {}

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------

            bool PriorFactor::active() const {
                return knot1_->getPose()->active() || knot1_->getVelocity()->active() ||
                       knot2_->getPose()->active() || knot2_->getVelocity()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void PriorFactor::getRelatedVarKeys(KeySet& keys) const {
                knot1_->getPose()->getRelatedVarKeys(keys);
                knot1_->getVelocity()->getRelatedVarKeys(keys);
                knot2_->getPose()->getRelatedVarKeys(keys);
                knot2_->getVelocity()->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto PriorFactor::value() const -> OutType {
                OutType error = OutType::Zero();

                const auto T1 = knot1_->getPose()->value();
                const auto w1 = knot1_->getVelocity()->value();
                const auto T2 = knot2_->getPose()->value();
                const auto w2 = knot2_->getVelocity()->value();

                const double dt = (knot2_->getTime() - knot1_->getTime()).seconds();
                const auto xi_21 = (T2 / T1).vec();
                
                // Compute prior error
                error.block<6, 1>(0, 0) = xi_21 - dt * w1;
                error.block<6, 1>(6, 0) = slam::liemath::se3::vec2jacinv(xi_21) * w2 - w1;
                
                return error;
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto PriorFactor::forward() const -> slam::eval::Node<OutType>::Ptr {
                const auto T1 = knot1_->getPose()->forward();
                const auto w1 = knot1_->getVelocity()->forward();
                const auto T2 = knot2_->getPose()->forward();
                const auto w2 = knot2_->getVelocity()->forward();

                const double dt = (knot2_->getTime() - knot1_->getTime()).seconds();
                const auto xi_21 = (T2->value() / T1->value()).vec();
                
                OutType error = OutType::Zero();
                error.block<6, 1>(0, 0) = xi_21 - dt * w1->value();
                error.block<6, 1>(6, 0) = slam::liemath::se3::vec2jacinv(xi_21) * w2->value() - w1->value();

                auto node = slam::eval::Node<OutType>::MakeShared(error);
                node->addChild(T1);
                node->addChild(w1);
                node->addChild(T2);
                node->addChild(w2);
                
                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void PriorFactor::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                       const slam::eval::Node<OutType>::Ptr& node,
                                       slam::eval::StateKeyJacobians& jacs) const {
                if (!active()) return;

                // Compute Jacobians using state transition properties
                const auto Fk1 = getJacKnot1(knot1_, knot2_);
                const auto Ek = getJacKnot2(knot1_, knot2_);

                if (knot1_->getPose()->active()) {
                    const auto T1 = std::dynamic_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(0));
                    Eigen::MatrixXd new_lhs = lhs * Fk1.block<12, 6>(0, 0);
                    knot1_->getPose()->backward(new_lhs, T1, jacs);
                }
                if (knot1_->getVelocity()->active()) {
                    const auto w1 = std::dynamic_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(1));
                    Eigen::MatrixXd new_lhs = lhs * Fk1.block<12, 6>(0, 6);
                    knot1_->getVelocity()->backward(new_lhs, w1, jacs);
                }
                if (knot2_->getPose()->active()) {
                    const auto T2 = std::dynamic_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(2));
                    Eigen::MatrixXd new_lhs = lhs * Ek.block<12, 6>(0, 0);
                    knot2_->getPose()->backward(new_lhs, T2, jacs);
                }
                if (knot2_->getVelocity()->active()) {
                    const auto w2 = std::dynamic_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(3));
                    Eigen::MatrixXd new_lhs = lhs * Ek.block<12, 6>(0, 6);
                    knot2_->getVelocity()->backward(new_lhs, w2, jacs);
                }
            }

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
