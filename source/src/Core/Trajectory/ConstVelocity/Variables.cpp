#include "Core/Trajectory/ConstVelocity/Variables.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            Variable::Ptr Variable::MakeShared(const slam::traj::Time& time,
                                               const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                                               const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink) {
                return std::make_shared<Variable>(time, T_k0, w_0k_ink);
            }

            // -----------------------------------------------------------------------------
            // Variable
            // -----------------------------------------------------------------------------

            Variable::Variable(const slam::traj::Time& time,
                               const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                               const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink)
                : time_(time), T_k0_(std::move(T_k0)), w_0k_ink_(std::move(w_0k_ink)) {}

            // -----------------------------------------------------------------------------
            // getTime
            // -----------------------------------------------------------------------------

            const slam::traj::Time& Variable::getTime() const {
                return time_;
            }

            // -----------------------------------------------------------------------------
            // getPose
            // -----------------------------------------------------------------------------

            const slam::eval::Evaluable<Variable::PoseType>::Ptr& Variable::getPose() const {
                return T_k0_;
            }

            // -----------------------------------------------------------------------------
            // getVelocity
            // -----------------------------------------------------------------------------

            const slam::eval::Evaluable<Variable::VelocityType>::Ptr& Variable::getVelocity() const {
                return w_0k_ink_;
            }

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
