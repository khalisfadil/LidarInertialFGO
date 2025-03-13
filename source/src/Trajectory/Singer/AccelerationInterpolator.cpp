#include "Trajectory/Singer/AccelerationInterpolator.hpp"

namespace slam {
    namespace traj {
        namespace singer {

            // -----------------------------------------------------------------------------
            // Factory Method
            // -----------------------------------------------------------------------------
            auto AccelerationInterpolator::MakeShared(const Time& time, 
                                                      const Variable::ConstPtr& knot1,
                                                      const Variable::ConstPtr& knot2,
                                                      const Eigen::Matrix<double, 6, 1>& ad) -> Ptr {
                return std::make_shared<AccelerationInterpolator>(time, knot1, knot2, ad);
            }
                    
            // -----------------------------------------------------------------------------
            // Constructor
            // -----------------------------------------------------------------------------
            AccelerationInterpolator::AccelerationInterpolator(const Time& time, 
                                                               const Variable::ConstPtr& knot1,
                                                               const Variable::ConstPtr& knot2,
                                                               const Eigen::Matrix<double, 6, 1>& ad)
                : slam::traj::const_acc::AccelerationInterpolator(time, knot1, knot2) {
                
                // Compute time intervals
                const double T = (knot2_->getTime() - knot1_->getTime()).seconds();
                const double tau = (time - knot1_->getTime()).seconds();
                const double kappa = (knot2_->getTime() - time).seconds();

                // Precompute covariance and transition matrices
                const auto Q_T_inv = getQ(T, ad).inverse();
                omega_ = getQ(tau, ad) * getTran(kappa, ad).transpose() * Q_T_inv;
                lambda_ = getTran(tau, ad) - omega_ * getTran(T, ad);
            }

        }  // namespace singer
    }  // namespace traj
}  // namespace slam
