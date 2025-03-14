#include "Core/Trajectory/ConstAcceleration/PriorFactor.hpp"
#include "Core/Trajectory/ConstAcceleration/Helper.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto PriorFactor::MakeShared(const Variable::ConstPtr& knot1,
                                         const Variable::ConstPtr& knot2) -> Ptr {
                return std::make_shared<PriorFactor>(knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // PriorFactor Constructor
            // -----------------------------------------------------------------------------

            PriorFactor::PriorFactor(const Variable::ConstPtr& knot1,
                                     const Variable::ConstPtr& knot2)
                : knot1_(knot1), knot2_(knot2) {
                const double dt = (knot2_->getTime() - knot1_->getTime()).seconds();
                Phi_ = getTran(dt);
            }

            // -----------------------------------------------------------------------------
            // Active
            // -----------------------------------------------------------------------------

            bool PriorFactor::active() const {
                return knot1_->getPose()->active() || knot1_->getVelocity()->active() ||
                       knot1_->getAcceleration()->active() || knot2_->getPose()->active() ||
                       knot2_->getVelocity()->active() || knot2_->getAcceleration()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void PriorFactor::getRelatedVarKeys(KeySet& keys) const {
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

            auto PriorFactor::value() const -> OutType {
                // Retrieve state values
                const auto T1 = knot1_->getPose()->value(), T2 = knot2_->getPose()->value();
                const auto w1 = knot1_->getVelocity()->value(), dw1 = knot1_->getAcceleration()->value();
                const auto w2 = knot2_->getVelocity()->value(), dw2 = knot2_->getAcceleration()->value();

                // Compute se(3) transformation and Jacobian
                const auto xi_21 = (T2 / T1).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Construct gamma1 and gamma2
                Eigen::Matrix<double, 18, 1> gamma1 = Eigen::Matrix<double, 18, 1>::Zero();
                gamma1.segment<6>(6) = w1;
                gamma1.segment<6>(12) = dw1;

                Eigen::Matrix<double, 18, 1> gamma2 = Eigen::Matrix<double, 18, 1>::Zero();
                gamma2.head<6>() = xi_21;
                gamma2.segment<6>(6) = J_21_inv * w2;
                gamma2.segment<6>(12) = -0.5 * liemath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2;

                return gamma2 - Phi_ * gamma1;
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto PriorFactor::forward() const -> slam::eval::Node<OutType>::Ptr {
                // Retrieve forward values
                const auto T1 = knot1_->getPose()->forward(), T2 = knot2_->getPose()->forward();
                const auto w1 = knot1_->getVelocity()->forward(), dw1 = knot1_->getAcceleration()->forward();
                const auto w2 = knot2_->getVelocity()->forward(), dw2 = knot2_->getAcceleration()->forward();

                // Compute se(3) transformation and Jacobian
                const auto xi_21 = (T2->value() / T1->value()).vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Construct gamma1 and gamma2
                Eigen::Matrix<double, 18, 1> gamma1 = Eigen::Matrix<double, 18, 1>::Zero();
                gamma1.segment<6>(6) = w1->value();
                gamma1.segment<6>(12) = dw1->value();

                Eigen::Matrix<double, 18, 1> gamma2 = Eigen::Matrix<double, 18, 1>::Zero();
                gamma2.head<6>() = xi_21;
                gamma2.segment<6>(6) = J_21_inv * w2->value();
                gamma2.segment<6>(12) = -0.5 * liemath::se3::curlyhat(J_21_inv * w2->value()) * w2->value() + J_21_inv * dw2->value();

                // Compute the error
                const auto node = eval::Node<OutType>::MakeShared(gamma2 - Phi_ * gamma1);

                // Explicitly specify the container type
                std::initializer_list<slam::eval::NodeBase::Ptr> children = {T1, w1, dw1, T2, w2, dw2};
                for (const auto& child : children) node->addChild(child);
                
                return node;
            }

            // -----------------------------------------------------------------------------
            // backward (Corrected)
            // -----------------------------------------------------------------------------

            void PriorFactor::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                       const eval::Node<OutType>::Ptr& node,
                                       slam::eval::StateKeyJacobians& jacs) const {
                // Knots and their Jacobian functions
                std::array<std::pair<const Variable::ConstPtr&, std::function<Eigen::Matrix<double, 18, 18>()>>, 2> knots = {{
                    {knot1_, [&] { return getJacKnot1_(); }},
                    {knot2_, [&] { return getJacKnot2_(); }}
                }};

                for (int knot_idx = 0; knot_idx < 2; ++knot_idx) {
                    const auto& knot = knots[knot_idx].first;
                    if (knot->getPose()->active() || knot->getVelocity()->active() || knot->getAcceleration()->active()) {
                        const auto J_knot = knots[knot_idx].second();

                        // Extract 18x6 blocks for pose, velocity, and acceleration
                        const auto J_pose = J_knot.block<18, 6>(0, 0);
                        const auto J_velocity = J_knot.block<18, 6>(0, 6);
                        const auto J_acceleration = J_knot.block<18, 6>(0, 12);

                        // Compute lhs * J_block separately for each
                        const auto lhs_J_pose = lhs * J_pose;
                        const auto lhs_J_velocity = lhs * J_velocity;
                        const auto lhs_J_acceleration = lhs * J_acceleration;

                        // Update Jacobians using lambda functions
                        std::array<std::pair<int, std::function<void()>>, 3> jacobian_updates = {{
                            {0, [&] {
                                if (knot->getPose()->active()) {
                                    auto T = std::static_pointer_cast<slam::eval::Node<InPoseType>>(node->getChild(knot_idx * 3));
                                    knot->getPose()->backward(lhs_J_pose, T, jacs);
                                }
                            }},
                            {1, [&] {
                                if (knot->getVelocity()->active()) {
                                    auto w = std::static_pointer_cast<slam::eval::Node<InVelType>>(node->getChild(knot_idx * 3 + 1));
                                    knot->getVelocity()->backward(lhs_J_velocity, w, jacs);
                                }
                            }},
                            {2, [&] {
                                if (knot->getAcceleration()->active()) {
                                    auto dw = std::static_pointer_cast<slam::eval::Node<InAccType>>(node->getChild(knot_idx * 3 + 2));
                                    knot->getAcceleration()->backward(lhs_J_acceleration, dw, jacs);
                                }
                            }}
                        }};

                        for (const auto& update : jacobian_updates) update.second();
                    }
                }
            }

            // -----------------------------------------------------------------------------
            // getJacKnot1_
            // -----------------------------------------------------------------------------

            Eigen::Matrix<double, 18, 18> PriorFactor::getJacKnot1_() const {
                return getJacKnot1(knot1_, knot2_);
            }

            // -----------------------------------------------------------------------------
            // getJacKnot2_
            // -----------------------------------------------------------------------------

            Eigen::Matrix<double, 18, 18> PriorFactor::getJacKnot2_() const {
                return getJacKnot2(knot1_, knot2_);
            }

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam