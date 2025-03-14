#include "Core/Trajectory/Singer/AccelerationExtrapolator.hpp"

namespace slam {
    namespace traj {
        namespace singer {

            // -----------------------------------------------------------------------------
            // VelocityExtrapolator
            // -----------------------------------------------------------------------------

            auto AccelerationExtrapolator::MakeShared(const Time& time, const Variable::ConstPtr& knot,
                                            const Eigen::Matrix<double, 6, 1>& ad)  -> Ptr {
                return std::make_shared<AccelerationExtrapolator>(time, knot, ad);
            }
                    
            // -----------------------------------------------------------------------------
            // VelocityExtrapolator
            // -----------------------------------------------------------------------------   

            AccelerationExtrapolator::AccelerationExtrapolator(const Time& time, const Variable::ConstPtr& knot,
                                            const Eigen::Matrix<double, 6, 1>& ad)
                    : slam::traj::const_acc::AccelerationExtrapolator(time, knot) {
                const double tau = (time - knot->getTime()).seconds();
                Phi_ = getTran(tau, ad);
            }

        }  // namespace singer
    }  // namespace traj
}  // namespace slam