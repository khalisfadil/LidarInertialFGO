#include "Core/Trajectory/Singer/PoseInterpolator.hpp"

namespace slam {
    namespace traj {
        namespace singer {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto PoseInterpolator::MakeShared(const Time& time, const Variable::ConstPtr& knot1,
                                                const Variable::ConstPtr& knot2,
                                                const Eigen::Matrix<double, 6, 1>& ad) -> Ptr {
                return std::make_shared<PoseInterpolator>(time, knot1, knot2, ad);
            }

            // -----------------------------------------------------------------------------
            // PoseInterpolator
            // -----------------------------------------------------------------------------

            PoseInterpolator::PoseInterpolator(const Time& time, const Variable::ConstPtr& knot1,
                                   const Variable::ConstPtr& knot2,
                                   const Eigen::Matrix<double, 6, 1>& ad)
                : slam::traj::const_acc::PoseInterpolator(time, knot1, knot2) {

                // Compute time constants
                const double T = (knot2->getTime() - knot1->getTime()).seconds();
                const double tau = (time - knot1->getTime()).seconds();
                const double kappa = (knot2->getTime() - time).seconds();

                // Precompute matrices
                const auto Q_T_inv = getQ(T, ad).inverse();
                
                // Compute interpolation values efficiently
                omega_ = getQ(tau, ad) * getTran(kappa, ad).transpose() * Q_T_inv;
                lambda_ = getTran(tau, ad) - omega_ * getTran(T, ad);
            }
        }  // namespace singer
    }  // namespace traj
}  // namespace slam