#include "Trajectory/Singer/Interface.hpp"

namespace slam {
    namespace traj {
        namespace singer {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto Interface::MakeShared(const Eigen::Matrix<double, 6, 1>& alpha_diag,
                                       const Eigen::Matrix<double, 6, 1>& Qc_diag) -> Ptr {
                return std::make_shared<Interface>(alpha_diag, Qc_diag);
            }

            // -----------------------------------------------------------------------------
            // Constructor
            // -----------------------------------------------------------------------------

            Interface::Interface(const Eigen::Matrix<double, 6, 1>& alpha_diag,
                                 const Eigen::Matrix<double, 6, 1>& Qc_diag)
                : slam::traj::const_acc::Interface(Qc_diag), alpha_diag_(alpha_diag) {}

            // -----------------------------------------------------------------------------
            // getJacKnot1_
            // -----------------------------------------------------------------------------

            inline Eigen::Matrix<double, 18, 18> Interface::getJacKnot1_(
                const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
                return getJacKnot1(knot1, knot2, alpha_diag_);
            }

            // -----------------------------------------------------------------------------
            // getQ_ / getQinv_
            // -----------------------------------------------------------------------------

            inline Eigen::Matrix<double, 18, 18> Interface::getQ_(
                const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
                return getQ(dt, alpha_diag_, Qc_diag);
            }

            inline Eigen::Matrix<double, 18, 18> Interface::getQinv_(
                const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
                return getQ(dt, alpha_diag_, Qc_diag).inverse();
            }

            // -----------------------------------------------------------------------------
            // getPoseInterpolator_
            // -----------------------------------------------------------------------------

            inline auto Interface::getPoseInterpolator_(
                const Time& time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const 
                -> slam::eval::Evaluable<PoseType>::Ptr {
                return PoseInterpolator::MakeShared(time, knot1, knot2, alpha_diag_);
            }

            // -----------------------------------------------------------------------------
            // getVelocityInterpolator_
            // -----------------------------------------------------------------------------

            inline auto Interface::getVelocityInterpolator_(
                const Time& time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const 
                -> slam::eval::Evaluable<VelocityType>::Ptr {
                return VelocityInterpolator::MakeShared(time, knot1, knot2, alpha_diag_);
            }

            // -----------------------------------------------------------------------------
            // getAccelerationInterpolator_
            // -----------------------------------------------------------------------------

            inline auto Interface::getAccelerationInterpolator_(
                const Time& time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const 
                -> slam::eval::Evaluable<AccelerationType>::Ptr {
                return AccelerationInterpolator::MakeShared(time, knot1, knot2, alpha_diag_);
            }

            // -----------------------------------------------------------------------------
            // getPoseExtrapolator_
            // -----------------------------------------------------------------------------

            inline auto Interface::getPoseExtrapolator_(
                const Time& time, const Variable::ConstPtr& knot) const 
                -> slam::eval::Evaluable<PoseType>::Ptr {
                return PoseExtrapolator::MakeShared(time, knot, alpha_diag_);
            }

            // -----------------------------------------------------------------------------
            // getVelocityExtrapolator_
            // -----------------------------------------------------------------------------

            inline auto Interface::getVelocityExtrapolator_(
                const Time& time, const Variable::ConstPtr& knot) const 
                -> slam::eval::Evaluable<VelocityType>::Ptr {
                return VelocityExtrapolator::MakeShared(time, knot, alpha_diag_);
            }

            // -----------------------------------------------------------------------------
            // getAccelerationExtrapolator_
            // -----------------------------------------------------------------------------

            inline auto Interface::getAccelerationExtrapolator_(
                const Time& time, const Variable::ConstPtr& knot) const 
                -> slam::eval::Evaluable<AccelerationType>::Ptr {
                return AccelerationExtrapolator::MakeShared(time, knot, alpha_diag_);
            }

            // -----------------------------------------------------------------------------
            // getPriorFactor_
            // -----------------------------------------------------------------------------

            inline auto Interface::getPriorFactor_(
                const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const 
                -> slam::eval::Evaluable<Eigen::Matrix<double, 18, 1>>::Ptr {
                return PriorFactor::MakeShared(knot1, knot2, alpha_diag_);
            }

            // -----------------------------------------------------------------------------
            // getQinvPublic
            // -----------------------------------------------------------------------------

            inline Eigen::Matrix<double, 18, 18> Interface::getQinvPublic(
                const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
                return getQ(dt, alpha_diag_, Qc_diag).inverse();
            }

            // -----------------------------------------------------------------------------
            // getQPublic
            // -----------------------------------------------------------------------------

            inline Eigen::Matrix<double, 18, 18> Interface::getQPublic(
                const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
                return getQ(dt, alpha_diag_, Qc_diag);
            }

            // -----------------------------------------------------------------------------
            // getQinvPublic (Overloaded)
            // -----------------------------------------------------------------------------

            inline Eigen::Matrix<double, 18, 18> Interface::getQinvPublic(const double& dt) const {
                return getQ(dt, alpha_diag_, Qc_diag_).inverse();
            }

            // -----------------------------------------------------------------------------
            // getQPublic (Overloaded)
            // -----------------------------------------------------------------------------

            inline Eigen::Matrix<double, 18, 18> Interface::getQPublic(const double& dt) const {
                return getQ(dt, alpha_diag_, Qc_diag_);
            }

            // -----------------------------------------------------------------------------
            // getTranPublic
            // -----------------------------------------------------------------------------
            
            inline Eigen::Matrix<double, 18, 18> Interface::getTranPublic(const double& dt) const {
                return getTran(dt, alpha_diag_);
            }

        }  // namespace singer
    }  // namespace traj
}  // namespace slam