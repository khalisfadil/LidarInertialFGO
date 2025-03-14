#include "Core/Trajectory/Singer/VelocityExtrapolator.hpp"

namespace slam {
    namespace traj {
        namespace singer {

            // -----------------------------------------------------------------------------
            // VelocityExtrapolator
            // -----------------------------------------------------------------------------

            auto VelocityExtrapolator::MakeShared(const Time& time, const Variable::ConstPtr& knot,
                                            const Eigen::Matrix<double, 6, 1>& ad)  -> Ptr {
                return std::make_shared<VelocityExtrapolator>(time, knot, ad);
            }
                    
            // -----------------------------------------------------------------------------
            // VelocityExtrapolator
            // -----------------------------------------------------------------------------   

            VelocityExtrapolator::VelocityExtrapolator(const Time& time, const Variable::ConstPtr& knot,
                                            const Eigen::Matrix<double, 6, 1>& ad)
                    : slam::traj::const_acc::VelocityExtrapolator(time, knot) {
                const double tau = (time - knot->getTime()).seconds();
                Phi_ = getTran(tau, ad);
            }

        }  // namespace singer
    }  // namespace traj
}  // namespace slam