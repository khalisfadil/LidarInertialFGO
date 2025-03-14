#include "Core/Trajectory/Singer/VelocityInterpolator.hpp"

namespace slam {
    namespace traj {
        namespace singer {

            // -----------------------------------------------------------------------------
            // Factory Method
            // -----------------------------------------------------------------------------
            auto VelocityInterpolator::MakeShared(const Time& time, 
                                                  const Variable::ConstPtr& knot1,
                                                  const Variable::ConstPtr& knot2,
                                                  const Eigen::Matrix<double, 6, 1>& ad) -> Ptr {
                return std::make_shared<VelocityInterpolator>(time, knot1, knot2, ad);
            }
                    
            // -----------------------------------------------------------------------------
            // Constructor
            // -----------------------------------------------------------------------------
            VelocityInterpolator::VelocityInterpolator(const Time& time, 
                                                       const Variable::ConstPtr& knot1,
                                                       const Variable::ConstPtr& knot2,
                                                       const Eigen::Matrix<double, 6, 1>& ad)
                : slam::traj::const_acc::VelocityInterpolator(time, knot1, knot2) {
                
                // Compute time intervals
                const double T = (knot2_->getTime() - knot1_->getTime()).seconds();
                const double tau = (time - knot1_->getTime()).seconds();
                const double kappa = (knot2_->getTime() - time).seconds();

                // Validate time intervals
                if (T <= 0 || tau < 0 || kappa < 0) {
                    throw std::invalid_argument("Time intervals must be positive: T=" + std::to_string(T) + 
                                                ", tau=" + std::to_string(tau) + 
                                                ", kappa=" + std::to_string(kappa));
                }

                // Precompute covariance and transition matrices with intermediate steps
                const auto Q_T = getQ(T, ad);                         // 18x18 process noise covariance for T
                if (Q_T.determinant() == 0) {                         // Check invertibility
                    throw std::runtime_error("Q_T is singular and cannot be inverted");
                }
                const auto Q_T_inv = Q_T.inverse();                   // Inverse of Q_T (18x18)
                const auto Q_tau = getQ(tau, ad);                     // 18x18 covariance for tau
                const auto Tran_kappa_T = getTran(kappa, ad).transpose(); // 18x18 transposed transition matrix
                const auto temp_omega = Q_tau * Tran_kappa_T;         // Intermediate: 18x18 * 18x18 = 18x18
                omega_ = temp_omega * Q_T_inv;                        // Final omega_: 18x18 * 18x18 = 18x18

                const auto Tran_tau = getTran(tau, ad);               // 18x18 transition matrix for tau
                const auto Tran_T = getTran(T, ad);                   // 18x18 transition matrix for T
                const auto temp_lambda = omega_ * Tran_T;             // Intermediate: 18x18 * 18x18 = 18x18
                lambda_ = Tran_tau - temp_lambda;                     // Final lambda_: 18x18 - 18x18 = 18x18
            }

        }  // namespace singer
    }  // namespace traj
}  // namespace slam