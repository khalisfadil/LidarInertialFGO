#include "source/include/Trajectory/Singer/PoseExtrapolator.hpp"


namespace slam {
    namespace traj {
        namespace singer {

            // -----------------------------------------------------------------------------
            // PoseInterpolator
            // -----------------------------------------------------------------------------

            auto PoseExtrapolator::MakeShared(const Time& time, const Variable::ConstPtr& knot,
                            const Eigen::Matrix<double, 6, 1>& ad)  -> Ptr {
                return std::make_shared<PoseExtrapolator>(time, knot, ad);
            }
                    
            // -----------------------------------------------------------------------------
            // PoseInterpolator
            // -----------------------------------------------------------------------------   

            PoseExtrapolator::PoseExtrapolator(const Time& time, const Variable::ConstPtr& knot,
                    const Eigen::Matrix<double, 6, 1>& ad)
                    : slam::traj::const_acc::PoseExtrapolator(time, knot) {
                const double tau = (time - knot->getTime()).seconds();
                Phi_ = getTran(tau, ad);
            }

        }  // namespace singer
    }  // namespace traj
}  // namespace slam