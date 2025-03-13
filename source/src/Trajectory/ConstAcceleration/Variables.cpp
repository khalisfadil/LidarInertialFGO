#include "Trajectory/ConstAcceleration/Variables.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto Variable::MakeShared(const Time& time, 
                                    const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                                    const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink,
                                    const slam::eval::Evaluable<AccelerationType>::Ptr& dw_0k_ink) -> Ptr {
                return std::make_shared<Variable>(time, T_k0, w_0k_ink, dw_0k_ink);
            }

            // -----------------------------------------------------------------------------
            // Variable Constructor
            // -----------------------------------------------------------------------------

            Variable::Variable(const Time& time, 
                            const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                            const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink,
                            const slam::eval::Evaluable<AccelerationType>::Ptr& dw_0k_ink)
                : time_(time), T_k0_(T_k0), w_0k_ink_(w_0k_ink), dw_0k_ink_(dw_0k_ink) {}

            // -----------------------------------------------------------------------------
            // time
            // -----------------------------------------------------------------------------

            const Time& Variable::getTime() const {
                return time_;
            }

            // -----------------------------------------------------------------------------
            // pose
            // -----------------------------------------------------------------------------

            const slam::eval::Evaluable<Variable::PoseType>::Ptr& Variable::getPose() const {
                return T_k0_;
            }

            // -----------------------------------------------------------------------------
            // velocity
            // -----------------------------------------------------------------------------

            const slam::eval::Evaluable<Variable::VelocityType>::Ptr& Variable::getVelocity() const {
                return w_0k_ink_;
            }

            // -----------------------------------------------------------------------------
            // acceleration
            // -----------------------------------------------------------------------------

            const slam::eval::Evaluable<Variable::AccelerationType>::Ptr& Variable::getAcceleration() const {
                return dw_0k_ink_;
            }

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
