#include "source/include/Trajectory/ConstAcceleration/PriorFactor.hpp"
#include "source/include/Trajectory/ConstAcceleration/Helper.hpp"

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
                const double dt = (knot2_->time() - knot1_->time()).seconds();
                Phi_ = getTran(dt);
            }

            // -----------------------------------------------------------------------------
            // Active
            // -----------------------------------------------------------------------------

            bool PriorFactor::active() const {
                return knot1_->pose()->active() || knot1_->velocity()->active() ||
                       knot1_->acceleration()->active() || knot2_->pose()->active() ||
                       knot2_->velocity()->active() || knot2_->acceleration()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void PriorFactor::getRelatedVarKeys(KeySet& keys) const {
                knot1_->pose()->getRelatedVarKeys(keys);
                knot1_->velocity()->getRelatedVarKeys(keys);
                knot1_->acceleration()->getRelatedVarKeys(keys);
                knot2_->pose()->getRelatedVarKeys(keys);
                knot2_->velocity()->getRelatedVarKeys(keys);
                knot2_->acceleration()->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto PriorFactor::value() const -> OutType {
                OutType error = OutType::Zero();

                const auto T1 = knot1_->pose()->value();
                const auto w1 = knot1_->velocity()->value();
                const auto dw1 = knot1_->acceleration()->value();
                const auto T2 = knot2_->pose()->value();
                const auto w2 = knot2_->velocity()->value();
                const auto dw2 = knot2_->acceleration()->value();

                const auto xi_21 = (T2 / T1).vec();
                const auto J_21_inv = liemath::se3::vec2jacinv(xi_21);

                Eigen::Matrix<double, 18, 1> gamma1 = Eigen::Matrix<double, 18, 1>::Zero();
                gamma1.block<6, 1>(6, 0) = w1;
                gamma1.block<6, 1>(12, 0) = dw1;
                Eigen::Matrix<double, 18, 1> gamma2 = Eigen::Matrix<double, 18, 1>::Zero();
                gamma2.block<6, 1>(0, 0) = xi_21;
                gamma2.block<6, 1>(6, 0) = J_21_inv * w2;
                gamma2.block<6, 1>(12, 0) =
                    -0.5 * liemath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2;

                error = gamma2 - Phi_ * gamma1;
                return error;
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
