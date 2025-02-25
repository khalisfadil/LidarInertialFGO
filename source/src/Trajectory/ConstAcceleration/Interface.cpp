#include "source/include/Trajectory/ConstAcceleration/Interface.hpp"

#include "source/include/Evaluable/se3/Evaluables.hpp"
#include "source/include/Evaluable/vspace/Evaluables.hpp"
#include "source/include/Problem/LossFunc/LossFunc.hpp"
#include "source/include/Problem/NoiseModel/StaticNoiseModel.hpp"
#include "source/include/Trajectory/ConstAcceleration/AccelerationExtrapolator.hpp"
#include "source/include/Trajectory/ConstAcceleration/AccelerationInterpolator.hpp"
#include "source/include/Trajectory/ConstAcceleration/Helper.hpp"
#include "source/include/Trajectory/ConstAcceleration/PoseExtrapolator.hpp"
#include "source/include/Trajectory/ConstAcceleration/PoseInterpolator.hpp"
#include "source/include/Trajectory/ConstAcceleration/PriorFactor.hpp"
#include "source/include/Trajectory/ConstAcceleration/VelocityExtrapolator.hpp"
#include "source/include/Trajectory/ConstAcceleration/VelocityInterpolator.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto Interface::MakeShared(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& Qc_diag) -> Ptr {
                return std::make_shared<Interface>(Qc_diag);
            }

            // -----------------------------------------------------------------------------
            // Interface
            // -----------------------------------------------------------------------------

            Interface::Interface(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& Qc_diag)
            : Qc_diag_(Qc_diag) {}

            // -----------------------------------------------------------------------------
            // add
            // -----------------------------------------------------------------------------

            void Interface::add(const Time& time, 
                                const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                                const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink,
                                const slam::eval::Evaluable<VelocityType>::Ptr& dw_0k_ink) {
                // Null check for essential evaluables
                if (!T_k0 || !w_0k_ink) {
                    throw std::invalid_argument("[Interface::add] Pose or velocity evaluable is null.");
                }

                // Check for duplicate time in knot_map_
                if (knot_map_.count(time)) {
                    throw std::runtime_error("[Interface::add] Duplicate trajectory knot at time " +
                                            std::to_string(time.seconds()));
                }

                // Insert new Variable into knot_map_
                knot_map_.emplace(time, std::make_shared<Variable>(time, T_k0, w_0k_ink, dw_0k_ink));
            }

            // -----------------------------------------------------------------------------
            // get
            // -----------------------------------------------------------------------------
            
            Variable::ConstPtr Interface::get(const Time& time) const {
                if (auto it = knot_map_.find(time); it != knot_map_.end()) {
                    return it->second;
                }
                throw std::out_of_range("[Interface::get] No trajectory knot exists at time " + std::to_string(time.seconds()));
            }

            // -----------------------------------------------------------------------------
            // getPoseInterpolator
            // -----------------------------------------------------------------------------

            auto Interface::getPoseInterpolator(const Time& time) const -> slam::eval::Evaluable<PoseType>::ConstPtr {
                if (knot_map_.empty()) 
                    throw std::runtime_error("[Interface::getPoseInterpolator] Knot map is empty");

                auto it_upper = knot_map_.lower_bound(time);

                // If `time` is beyond the last entry, extrapolate from the last knot
                if (it_upper == knot_map_.end()) 
                    return getPoseExtrapolator_(time, std::prev(it_upper)->second);

                // If `time` matches exactly, return the existing pose evaluator
                if (it_upper->second->getTime() == time) 
                    return it_upper->second->getPose();

                // If `time` is before the first entry, extrapolate from the first knot
                if (it_upper == knot_map_.begin()) 
                    return getPoseExtrapolator_(time, it_upper->second);

                // Get iterators bounding the interpolation interval
                auto it_lower = std::prev(it_upper);
                
                if (time <= it_lower->second->getTime() || time >= it_upper->second->getTime()) 
                    throw std::runtime_error("[Interface::getPoseInterpolator] Requested interpolation at an invalid time: " +
                                            std::to_string(time.seconds()) + " not in (" +
                                            std::to_string(it_lower->second->getTime().seconds()) + ", " +
                                            std::to_string(it_upper->second->getTime().seconds()) + ")");

                return getPoseInterpolator_(time, it_lower->second, it_upper->second);
            }

            // -----------------------------------------------------------------------------
            // getVelocityInterpolator
            // -----------------------------------------------------------------------------

            auto Interface::getVelocityInterpolator(const Time& time) const -> slam::eval::Evaluable<VelocityType>::ConstPtr {
                if (knot_map_.empty()) 
                    throw std::runtime_error("[Interface::getVelocityInterpolator] Knot map is empty");

                auto it_upper = knot_map_.lower_bound(time);

                // If `time` is beyond the last entry, extrapolate from the last knot
                if (it_upper == knot_map_.end()) 
                    return getVelocityExtrapolator_(time, std::prev(it_upper)->second);

                // If `time` matches exactly, return the existing velocity evaluator
                if (it_upper->second->getTime() == time) 
                    return it_upper->second->getVelocity();

                // If `time` is before the first entry, extrapolate from the first knot
                if (it_upper == knot_map_.begin()) 
                    return getVelocityExtrapolator_(time, it_upper->second);

                // Get iterators bounding the interpolation interval
                auto it_lower = std::prev(it_upper);

                if (time <= it_lower->second->getTime() || time >= it_upper->second->getTime()) 
                    throw std::runtime_error("[Interface::getVelocityInterpolator] Requested interpolation at an invalid time: " +
                                            std::to_string(time.seconds()) + " not in (" +
                                            std::to_string(it_lower->second->getTime().seconds()) + ", " +
                                            std::to_string(it_upper->second->getTime().seconds()) + ")");

                return getVelocityInterpolator_(time, it_lower->second, it_upper->second);
            }

            // -----------------------------------------------------------------------------
            // getAccelerationInterpolator
            // -----------------------------------------------------------------------------

            auto Interface::getAccelerationInterpolator(const Time& time) const -> slam::eval::Evaluable<AccelerationType>::ConstPtr {
                if (knot_map_.empty()) 
                    throw std::runtime_error("[Interface::getAccelerationInterpolator] Knot map is empty");

                auto it_upper = knot_map_.lower_bound(time);

                // If `time` is beyond the last entry, extrapolate from the last knot
                if (it_upper == knot_map_.end()) 
                    return getAccelerationExtrapolator_(time, std::prev(it_upper)->second);

                // If `time` matches exactly, return the existing acceleration evaluator
                if (it_upper->second->getTime() == time) 
                    return it_upper->second->getAcceleration();

                // If `time` is before the first entry, extrapolate from the first knot
                if (it_upper == knot_map_.begin()) 
                    return getAccelerationExtrapolator_(time, it_upper->second);

                // Get iterators bounding the interpolation interval
                auto it_lower = std::prev(it_upper);

                if (time <= it_lower->second->getTime() || time >= it_upper->second->getTime()) 
                    throw std::runtime_error("[Interface::getAccelerationInterpolator] Requested interpolation at an invalid time: " +
                                            std::to_string(time.seconds()) + " not in (" +
                                            std::to_string(it_lower->second->getTime().seconds()) + ", " +
                                            std::to_string(it_upper->second->getTime().seconds()) + ")");

                return getAccelerationInterpolator_(time, it_lower->second, it_upper->second);
            }

            // -----------------------------------------------------------------------------
            // getCovariance
            // -----------------------------------------------------------------------------

            auto Interface::getCovariance(const slam::solver::Covariance& cov, const Time& time) -> CovType {
                if (knot_map_.empty()) 
                    throw std::runtime_error("[Interface::getCovariance] Knot map is empty");

                auto it_upper = knot_map_.lower_bound(time);

                // Extrapolate after last entry
                if (it_upper == knot_map_.end()) {
                    const auto& endKnot = std::prev(it_upper)->second;

                    // Ensure active state variables
                    if (!endKnot->getPose()->active() || !endKnot->getVelocity()->active() || !endKnot->getAcceleration()->active())
                        throw std::runtime_error("[Interface::getCovariance] Extrapolation from a locked knot not implemented.");

                    // Convert to state variables
                    auto T_k0_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(endKnot->getPose());
                    auto w_0k_ink_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(endKnot->getVelocity());
                    auto dw_0k_ink_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(endKnot->getAcceleration());

                    if (!T_k0_var || !w_0k_ink_var || !dw_0k_ink_var)
                        throw std::runtime_error("[Interface::getCovariance] Trajectory states are not variables.");

                    // Construct extrapolated knot
                    auto extrap_knot = Variable::MakeShared(time, 
                                                            getPoseExtrapolator_(time, endKnot),
                                                            getVelocityExtrapolator_(time, endKnot),
                                                            getAccelerationExtrapolator_(time, endKnot));

                    // Compute Jacobians
                    auto F_t1 = -getJacKnot1_(endKnot, extrap_knot);
                    auto E_t1_inv = getJacKnot2_(endKnot, extrap_knot).inverse();

                    // Compute prior covariance
                    auto Qt1 = getQ_((extrap_knot->getTime() - endKnot->getTime()).seconds(), Qc_diag_);
                    auto P_end = cov.query({T_k0_var, w_0k_ink_var, dw_0k_ink_var});

                    // Compute covariance
                    return E_t1_inv * (F_t1 * P_end * F_t1.transpose() + Qt1) * E_t1_inv.transpose();
                }

                // If `time` matches exactly, return stored covariance
                if (it_upper->second->getTime() == time) {
                    const auto& knot = it_upper->second;

                    // Ensure active state variables
                    if (!knot->getPose()->active() || !knot->getVelocity()->active() || !knot->getAcceleration()->active())
                        throw std::runtime_error("[Interface::getCovariance] Extrapolation from a locked knot not implemented.");

                    auto T_k0_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(knot->getPose());
                    auto w_0k_ink_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(knot->getVelocity());
                    auto dw_0k_ink_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(knot->getAcceleration());

                    if (!T_k0_var || !w_0k_ink_var || !dw_0k_ink_var)
                        throw std::runtime_error("[Interface::getCovariance] Trajectory states are not variables.");

                    return cov.query({T_k0_var, w_0k_ink_var, dw_0k_ink_var});
                }

                // If `time` is before the first entry, throw an error
                if (it_upper == knot_map_.begin())
                    throw std::runtime_error("[Interface::getCovariance] Requested covariance before first time.");

                // Get iterators bounding the interpolation interval
                auto it_lower = std::prev(it_upper);
                const auto& knot1 = it_lower->second;
                const auto& knot2 = it_upper->second;

                // Ensure active state variables
                if (!knot1->getPose()->active() || !knot1->getVelocity()->active() || !knot1->getAcceleration()->active() ||
                    !knot2->getPose()->active() || !knot2->getVelocity()->active() || !knot2->getAcceleration()->active()) {
                    throw std::runtime_error("[Interface::getCovariance] Extrapolation from a locked knot not implemented.");
                }

                // Convert to state variables
                auto T_10_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(knot1->getPose());
                auto w_01_in1_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(knot1->getVelocity());
                auto dw_01_in1_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(knot1->getAcceleration());
                auto T_20_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(knot2->getPose());
                auto w_02_in2_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(knot2->getVelocity());
                auto dw_02_in2_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(knot2->getAcceleration());

                if (!T_10_var || !w_01_in1_var || !dw_01_in1_var || !T_20_var || !w_02_in2_var || !dw_02_in2_var)
                    throw std::runtime_error("[Interface::getCovariance] Trajectory states are not variables.");

                // Construct interpolated knot
                auto knotq = Variable::MakeShared(time, 
                                                getPoseInterpolator_(time, knot1, knot2),
                                                getVelocityInterpolator_(time, knot1, knot2),
                                                getAccelerationInterpolator_(time, knot1, knot2));

                // Compute Jacobians
                auto F_t1 = -getJacKnot1_(knot1, knotq);
                auto E_t1 = getJacKnot2_(knot1, knotq);
                auto F_2t = -getJacKnot1_(knotq, knot2);
                auto E_2t = getJacKnot2_(knotq, knot2);

                // Compute inverse prior covariances
                auto Qt1_inv = getQinv_((knotq->getTime() - knot1->getTime()).seconds(), Qc_diag_);
                auto Q2t_inv = getQinv_((knot2->getTime() - knotq->getTime()).seconds(), Qc_diag_);

                // Query covariance of knot1 and knot2
                auto P_1n2 = cov.query({T_10_var, w_01_in1_var, dw_01_in1_var, T_20_var, w_02_in2_var, dw_02_in2_var});

                // Helper matrices
                Eigen::Matrix<double, 36, 18> A;
                A << F_t1.transpose() * Qt1_inv * E_t1, 
                    E_2t.transpose() * Q2t_inv * F_2t;

                Eigen::Matrix<double, 36, 36> B = Eigen::Matrix<double, 36, 36>::Zero();
                B.block<18, 18>(0, 0) = F_t1.transpose() * Qt1_inv * F_t1;
                B.block<18, 18>(18, 18) = E_2t.transpose() * Q2t_inv * E_2t;

                // Compute interpolated covariance
                auto P_t_inv = E_t1.transpose() * Qt1_inv * E_t1 + 
                            F_2t.transpose() * Q2t_inv * F_2t -
                            A.transpose() * (P_1n2.inverse() + B).inverse() * A;

                return P_t_inv.inverse();
            }

            // -----------------------------------------------------------------------------
            // addPosePrior
            // -----------------------------------------------------------------------------

            void Interface::addPosePrior(const Time& time, const PoseType& T_k0,
                             const Eigen::Matrix<double, 6, 6>& cov) {
                if (pose_prior_factor_) 
                    throw std::runtime_error("[Interface::addPosePrior] Can only add one pose prior.");

                if (knot_map_.empty()) 
                    throw std::runtime_error("[Interface::addPosePrior] Knot map is empty.");

                auto it = knot_map_.find(time);
                if (it == knot_map_.end()) 
                    throw std::runtime_error("[Interface::addPosePrior] No knot at provided time.");

                const auto& knot = it->second;

                if (!knot->getPose()->active()) 
                    throw std::runtime_error("[Interface::addPosePrior] Attempted to add prior to a locked pose.");

                // Create cost term using streamlined initialization
                pose_prior_factor_ = slam::problem::costterm::WeightedLeastSqCostTerm<6>::MakeShared(
                    slam::eval::se3::se3_error(knot->getPose(), T_k0), 
                    slam::problem::noisemodel::StaticNoiseModel<6>::MakeShared(cov), 
                    slam::problem::lossfunc::L2LossFunc::MakeShared());
            }

            // -----------------------------------------------------------------------------
            // addVelocityPrior
            // -----------------------------------------------------------------------------

            void Interface::addVelocityPrior(const Time& time, const VelocityType& w_0k_ink,
                                 const Eigen::Matrix<double, 6, 6>& cov) {
                if (vel_prior_factor_) 
                    throw std::runtime_error("[Interface::addVelocityPrior] Can only add one velocity prior.");

                if (knot_map_.empty()) 
                    throw std::runtime_error("[Interface::addVelocityPrior] Knot map is empty.");

                auto it = knot_map_.find(time);
                if (it == knot_map_.end()) 
                    throw std::runtime_error("[Interface::addVelocityPrior] No knot found at provided time.");

                const auto& knot = it->second;

                if (!knot->getVelocity()->active()) 
                    throw std::runtime_error("[Interface::addVelocityPrior] Attempted to add prior to a locked velocity.");

                // Directly initialize cost term with error function, noise model, and loss function
                vel_prior_factor_ = slam::problem::costterm::WeightedLeastSqCostTerm<6>::MakeShared(
                    slam::eval::vspace::vspace_error<6>(knot->getVelocity(), w_0k_ink),
                    slam::problem::noisemodel::StaticNoiseModel<6>::MakeShared(cov),
                    slam::problem::lossfunc::L2LossFunc::MakeShared());
            }

            // -----------------------------------------------------------------------------
            // addAccelerationPrior
            // -----------------------------------------------------------------------------

            void Interface::addAccelerationPrior(const Time& time,
                                     const AccelerationType& dw_0k_ink,
                                     const Eigen::Matrix<double, 6, 6>& cov) {
                if (acc_prior_factor_)
                    throw std::runtime_error("[Interface::addAccelerationPrior] Can only add one acceleration prior.");

                if (knot_map_.empty())
                    throw std::runtime_error("[Interface::addAccelerationPrior] Knot map is empty.");

                auto it = knot_map_.find(time);
                if (it == knot_map_.end())
                    throw std::runtime_error("[Interface::addAccelerationPrior] No knot found at provided time.");

                const auto& knot = it->second;

                if (!knot->getAcceleration()->active())
                    throw std::runtime_error("[Interface::addAccelerationPrior] Attempted to add prior to a locked acceleration.");

                // Directly initialize cost term with error function, noise model, and loss function
                acc_prior_factor_ = slam::problem::costterm::WeightedLeastSqCostTerm<6>::MakeShared(
                    slam::eval::vspace::vspace_error<6>(knot->getAcceleration(), dw_0k_ink),
                    slam::problem::noisemodel::StaticNoiseModel<6>::MakeShared(cov),
                    slam::problem::lossfunc::L2LossFunc::MakeShared());
            }

            // -----------------------------------------------------------------------------
            // addStatePrior
            // -----------------------------------------------------------------------------

            void Interface::addStatePrior(const Time& time, const PoseType& T_k0,
                              const VelocityType& w_0k_ink,
                              const AccelerationType& dw_0k_ink,
                              const CovType& cov) {
                if (state_prior_factor_)
                    throw std::runtime_error("[Interface::addStatePrior] Can only add one state prior.");

                if (pose_prior_factor_ || vel_prior_factor_ || acc_prior_factor_)
                    throw std::runtime_error("[Interface::addStatePrior] A pose, velocity, or acceleration prior already exists.");

                if (knot_map_.empty())
                    throw std::runtime_error("[Interface::addStatePrior] Knot map is empty.");

                auto it = knot_map_.find(time);
                if (it == knot_map_.end())
                    throw std::runtime_error("[Interface::addStatePrior] No knot found at provided time.");

                const auto& knot = it->second;

                if (!knot->getPose()->active() || !knot->getVelocity()->active() || !knot->getAcceleration()->active())
                    throw std::runtime_error("[Interface::addStatePrior] Attempted to add prior to a locked state.");

                // Create merged error functions directly in the cost term
                state_prior_factor_ = slam::problem::costterm::WeightedLeastSqCostTerm<18>::MakeShared(
                    slam::eval::vspace::merge<12, 6>(slam::eval::vspace::merge<6, 6>(
                        slam::eval::se3::se3_error(knot->getPose(), T_k0),
                        slam::eval::vspace::vspace_error<6>(knot->getVelocity(), w_0k_ink)),
                        slam::eval::vspace::vspace_error<6>(knot->getAcceleration(), dw_0k_ink)),
                    slam::problem::noisemodel::StaticNoiseModel<18>::MakeShared(cov),
                    slam::problem::lossfunc::L2LossFunc::MakeShared());
            }

            // -----------------------------------------------------------------------------
            // addPriorCostTerms
            // -----------------------------------------------------------------------------

            void Interface::addPriorCostTerms(slam::problem::Problem& problem) const {
                if (knot_map_.empty()) return;

                // Add available prior factors
                for (const auto& prior_factor : {pose_prior_factor_, vel_prior_factor_, acc_prior_factor_})
                    if (prior_factor) problem.addCostTerm(prior_factor);

                // Use a single shared L2 loss function
                static const auto loss_function = slam::problem::lossfunc::L2LossFunc::MakeShared();

                // Iterate over adjacent knots and add prior cost terms for active variables
                for (auto it1 = knot_map_.begin(), it2 = std::next(it1); it2 != knot_map_.end(); ++it1, ++it2) {
                    const auto& [_, knot1] = *it1;
                    const auto& [__, knot2] = *it2;

                    // Skip if all states are locked
                    if (!(knot1->getPose()->active() || knot1->getVelocity()->active() || knot1->getAcceleration()->active() ||
                        knot2->getPose()->active() || knot2->getVelocity()->active() || knot2->getAcceleration()->active()))
                        continue;

                    // Compute information matrix for GP prior factor
                    const auto Qinv = getQinv_((knot2->getTime() - knot1->getTime()).seconds(), Qc_diag_);
                    const auto noise_model = slam::problem::noisemodel::StaticNoiseModel<18>::MakeShared(Qinv, slam::problem::noisemodel::NoiseType::INFORMATION);

                    // Create and add cost term
                    problem.addCostTerm(slam::problem::costterm::WeightedLeastSqCostTerm<18>::MakeShared(
                        getPriorFactor_(knot1, knot2), noise_model, loss_function));
                }
            }

            // -----------------------------------------------------------------------------
            // getJacKnot1_
            // -----------------------------------------------------------------------------

            inline Eigen::Matrix<double, 18, 18> Interface::getJacKnot1_(
                const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
                return getJacKnot1(knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // getJacKnot2_
            // -----------------------------------------------------------------------------

            inline Eigen::Matrix<double, 18, 18> Interface::getJacKnot2_(
                const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
                return getJacKnot2(knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // getQ_
            // -----------------------------------------------------------------------------

            inline Eigen::Matrix<double, 18, 18> Interface::getQ_(
                const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
                return getQ(dt, Qc_diag);
            }

            // -----------------------------------------------------------------------------
            // getQinv_
            // -----------------------------------------------------------------------------

            inline Eigen::Matrix<double, 18, 18> Interface::getQinv_(
                const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
                return getQinv(dt, Qc_diag);
            }

            // -----------------------------------------------------------------------------
            // getPoseInterpolator_
            // -----------------------------------------------------------------------------

            inline auto Interface::getPoseInterpolator_(
                const Time& time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const -> slam::eval::Evaluable<PoseType>::Ptr {
                return PoseInterpolator::MakeShared(time, knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // getVelocityInterpolator_
            // -----------------------------------------------------------------------------

            inline auto Interface::getVelocityInterpolator_(
                const Time& time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const -> slam::eval::Evaluable<VelocityType>::Ptr {
                return VelocityInterpolator::MakeShared(time, knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // getAccelerationInterpolator_
            // -----------------------------------------------------------------------------

            inline auto Interface::getAccelerationInterpolator_(
                const Time& time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const -> slam::eval::Evaluable<AccelerationType>::Ptr {
                return AccelerationInterpolator::MakeShared(time, knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // getPoseExtrapolator_
            // -----------------------------------------------------------------------------

            inline auto Interface::getPoseExtrapolator_(
                const Time& time, const Variable::ConstPtr& knot) const -> slam::eval::Evaluable<PoseType>::Ptr {
                return PoseExtrapolator::MakeShared(time, std::move(knot));
            }

            // -----------------------------------------------------------------------------
            // getVelocityExtrapolator_
            // -----------------------------------------------------------------------------

            inline auto Interface::getVelocityExtrapolator_(
                const Time& time, const Variable::ConstPtr& knot) const -> slam::eval::Evaluable<VelocityType>::Ptr {
                return VelocityExtrapolator::MakeShared(time, std::move(knot));
            }

            // -----------------------------------------------------------------------------
            // getAccelerationExtrapolator_
            // -----------------------------------------------------------------------------

            inline auto Interface::getAccelerationExtrapolator_(
                const Time& time, const Variable::ConstPtr& knot) const -> slam::eval::Evaluable<AccelerationType>::Ptr {
                return AccelerationExtrapolator::MakeShared(time, std::move(knot));
            }

            // -----------------------------------------------------------------------------
            // getQinvPublic
            // -----------------------------------------------------------------------------
            
            inline Eigen::Matrix<double, 18, 18> Interface::getQinvPublic(
                const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
                return getQinv(dt, Qc_diag);
            }

            // -----------------------------------------------------------------------------
            // getQinvPublic
            // -----------------------------------------------------------------------------

            inline Eigen::Matrix<double, 18, 18> Interface::getQinvPublic(const double& dt) const {
                return getQinv(dt, Qc_diag_);
            }

            // -----------------------------------------------------------------------------
            // getQPublic
            // -----------------------------------------------------------------------------

            inline Eigen::Matrix<double, 18, 18> Interface::getQPublic(const double& dt) const {
                return getQ(dt, Qc_diag_);
            }

            // -----------------------------------------------------------------------------
            // getQPublic
            // -----------------------------------------------------------------------------

            inline Eigen::Matrix<double, 18, 18> Interface::getQPublic(
                const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
                return getQ(dt, Qc_diag);
            }

            // -----------------------------------------------------------------------------
            // getTranPublic
            // -----------------------------------------------------------------------------

            inline Eigen::Matrix<double, 18, 18> Interface::getTranPublic(const double& dt) const {
                return getTran(dt);
            }

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam