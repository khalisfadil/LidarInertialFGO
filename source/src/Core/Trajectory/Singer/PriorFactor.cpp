#include "Core/Trajectory/Singer/PriorFactor.hpp"

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
                // Ensure knot pointers are valid before accessing
                assert(knot1_ && knot2_ && "Knot pointers must not be null");

                // Compute time interval in seconds
                const double dt = (knot2_->getTime() - knot1_->getTime()).seconds();

                // Compute state transition matrix Phi_ with damping parameters
                Phi_ = getTran(dt, ad);
            }

            // -----------------------------------------------------------------------------
            // Compute State Transition Jacobian for Knot1
            // -----------------------------------------------------------------------------

            Eigen::Matrix<double, 18, 18> PriorFactor::getJacKnot1_() const {
                // Ensure knot pointers are valid
                assert(knot1_ && knot2_ && "Knot pointers must not be null in getJacKnot1_");

                // Compute and return the Jacobian for the first knot
                return getJacKnot1(knot1_, knot2_, alpha_diag_);
            }

        }  // namespace singer
    }  // namespace traj
}  // namespace slam