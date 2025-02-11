#include "source/include/Trajectory/ConstAcceleration/VelocityExtrapolator.hpp"
#include "source/include/Trajectory/ConstAcceleration/Helper.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto VelocityExtrapolator::MakeShared(const Time time,
                                                const Variable::ConstPtr& knot) -> Ptr {
                return std::make_shared<VelocityExtrapolator>(time, knot);
            }

            // -----------------------------------------------------------------------------
            // Constructor
            // -----------------------------------------------------------------------------

            VelocityExtrapolator::VelocityExtrapolator(const Time time,
                                                    const Variable::ConstPtr& knot)
                : knot_(knot) {
                // Compute transition matrix for the given time shift
                const double tau = (time - knot->time()).seconds();
                Phi_ = getTran(tau);
            }

            // -----------------------------------------------------------------------------
            // Active Check
            // -----------------------------------------------------------------------------

            bool VelocityExtrapolator::active() const {
                return knot_->velocity()->active() || knot_->acceleration()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void VelocityExtrapolator::getRelatedVarKeys(eval::Evaluable<OutType>::KeySet& keys) const {
                knot_->velocity()->getRelatedVarKeys(keys);
                knot_->acceleration()->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto VelocityExtrapolator::value() const -> OutType {
                // Compute extrapolated velocity
                const Eigen::Matrix<double, 6, 1> xi_j1 =
                    Phi_.block<6, 6>(6, 6) * knot_->velocity()->value() +
                    Phi_.block<6, 6>(6, 12) * knot_->acceleration()->value();

                return xi_j1;  // Approximation holds as long as xi_i1 is small.
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto VelocityExtrapolator::forward() const -> eval::Node<OutType>::Ptr {
                // Forward propagate velocity and acceleration
                const auto w = knot_->velocity()->forward();
                const auto dw = knot_->acceleration()->forward();

                // Compute extrapolated velocity
                const Eigen::Matrix<double, 6, 1> xi_j1 =
                    Phi_.block<6, 6>(6, 6) * w->value() +
                    Phi_.block<6, 6>(6, 12) * dw->value();

                // Create computational node
                const auto node = eval::Node<OutType>::MakeShared(xi_j1);
                node->addChild(w);
                node->addChild(dw);

                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void VelocityExtrapolator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                                const eval::Node<OutType>::Ptr& node,
                                                eval::StateKeyJacobians& jacs) const {
                if (!active()) return;

                // Compute Jacobian for velocity term
                if (knot_->velocity()->active()) {
                    const auto w = std::static_pointer_cast<eval::Node<InVelType>>(node->getChild(0));
                    Eigen::MatrixXd new_lhs = lhs * Phi_.block<6, 6>(6, 6);
                    knot_->velocity()->backward(new_lhs, w, jacs);
                }

                // Compute Jacobian for acceleration term
                if (knot_->acceleration()->active()) {
                    const auto dw = std::static_pointer_cast<eval::Node<InAccType>>(node->getChild(1));
                    Eigen::MatrixXd new_lhs = lhs * Phi_.block<6, 6>(6, 12);
                    knot_->acceleration()->backward(new_lhs, dw, jacs);
                }
            }

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
