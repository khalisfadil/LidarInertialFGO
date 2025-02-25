#include "source/include/Trajectory/Singer/PriorFactor.hpp"

namespace slam {
    namespace traj {
        namespace singer {

            // -----------------------------------------------------------------------------
            // Factory Method
            // -----------------------------------------------------------------------------

            auto PriorFactor::MakeShared(const Variable::ConstPtr& knot1,
                                         const Variable::ConstPtr& knot2,
                                         const Eigen::Matrix<double, 6, 1>& ad) -> Ptr {
                return std::make_shared<PriorFactor>(knot1, knot2, ad);
            }
                    
            // -----------------------------------------------------------------------------
            // Constructor
            // -----------------------------------------------------------------------------

            PriorFactor::PriorFactor(const Variable::ConstPtr& knot1,
                                     const Variable::ConstPtr& knot2,
                                     const Eigen::Matrix<double, 6, 1>& ad)
                : slam::traj::const_acc::PriorFactor(knot1, knot2), alpha_diag_(ad) {

                // Compute **state transition matrix** Phi_ with damping parameters
                const double dt = (knot2_->getTime() - knot1_->getTime()).seconds();
                Phi_ = getTran(dt, ad);
            }

            // -----------------------------------------------------------------------------
            // Compute **State Transition Jacobian**
            // -----------------------------------------------------------------------------
            
            Eigen::Matrix<double, 18, 18> PriorFactor::getJacKnot1_() const {
                return getJacKnot1(knot1_, knot2_, alpha_diag_);
            }

        }  // namespace singer
    }  // namespace traj
}  // namespace slam
