#include "Trajectory/ConstVelocity/PriorFactor.hpp"

#include "Evaluable/se3/Evaluables.hpp"
#include "Evaluable/vspace/Evaluables.hpp"
#include "Trajectory/ConstVelocity/Helper.hpp"
#include "Trajectory/ConstVelocity/Evaluable/JinvVelocityEvaluator.hpp"

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

                error.block<6, 1>(0, 0) = xi_21 - dt * w1;
                error.block<6, 1>(6, 0) = liemath::se3::vec2jacinv(xi_21) * w2 - w1;

                return error;
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto PriorFactor::forward() const -> slam::eval::Node<OutType>::Ptr {
                // Retrieve state values
                const auto T1 = knot1_->getPose()->forward();
                const auto w1 = knot1_->getVelocity()->forward();
                const auto T2 = knot2_->getPose()->forward();
                const auto w2 = knot2_->getVelocity()->forward();

                // Precompute values
                const auto T1_val = T1->value();
                const auto w1_val = w1->value();
                const double dt = (knot2_->getTime() - knot1_->getTime()).seconds();
                const Eigen::Matrix<double, 6, 1> xi_21 = (T2->value() / T1_val).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Compute error directly
                OutType error = OutType::Zero();
                error.block<6, 1>(0, 0) = xi_21 - dt * w1_val;
                error.block<6, 1>(6, 0) = J_21_inv * w2->value() - w1_val;

                // Create node with error
                const auto node = slam::eval::Node<OutType>::MakeShared(error);
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
                // Early exit if no variables are active
                if (!active()) {
                    return;
                }

                // Precompute Jacobians
                const Eigen::MatrixXd Fk1 = (knot1_->getPose()->active() || knot1_->getVelocity()->active())
                                    ? Eigen::MatrixXd(getJacKnot1(knot1_, knot2_))
                                    : Eigen::MatrixXd::Zero(12, 12);
                const Eigen::MatrixXd Ek = (knot2_->getPose()->active() || knot2_->getVelocity()->active())
                                    ? Eigen::MatrixXd(getJacKnot2(knot1_, knot2_))
                                    : Eigen::MatrixXd::Zero(12, 12);

                // Lambda-based Jacobian updates with intermediate variables
                std::array<std::function<void()>, 4> updates = {
                    [&]() {
                        if (knot1_->getPose()->active()) {
                            const auto Fk1_block = Fk1.block<12, 6>(0, 0);     // Step 1: Extract block
                            const auto lhs_Fk1 = lhs * Fk1_block;              // Step 2: Compute product
                            const auto T1 = std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(0));
                            knot1_->getPose()->backward(lhs_Fk1, T1, jacs);    // Step 3: Pass result
                        }
                    },
                    [&]() {
                        if (knot1_->getVelocity()->active()) {
                            const auto Fk1_block = Fk1.block<12, 6>(0, 6);
                            const auto lhs_Fk1 = lhs * Fk1_block;
                            const auto w1 = std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(1));
                            knot1_->getVelocity()->backward(lhs_Fk1, w1, jacs);
                        }
                    },
                    [&]() {
                        if (knot2_->getPose()->active()) {
                            const auto Ek_block = Ek.block<12, 6>(0, 0);
                            const auto lhs_Ek = lhs * Ek_block;
                            const auto T2 = std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(2));
                            knot2_->getPose()->backward(lhs_Ek, T2, jacs);
                        }
                    },
                    [&]() {
                        if (knot2_->getVelocity()->active()) {
                            const auto Ek_block = Ek.block<12, 6>(0, 6);
                            const auto lhs_Ek = lhs * Ek_block;
                            const auto w2 = std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(3));
                            knot2_->getVelocity()->backward(lhs_Ek, w2, jacs);
                        }
                    }
                };

                // Execute updates
                for (const auto& update : updates) update();
            }

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam