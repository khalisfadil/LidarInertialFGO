#include "Trajectory/ConstAcceleration/Interface.hpp"

#include "Evaluable/se3/Evaluables.hpp"
#include "Evaluable/vspace/Evaluables.hpp"
#include "Problem/LossFunc/LossFunc.hpp"
#include "Problem/NoiseModel/StaticNoiseModel.hpp"
#include "Trajectory/ConstAcceleration/AccelerationExtrapolator.hpp"
#include "Trajectory/ConstAcceleration/AccelerationInterpolator.hpp"
#include "Trajectory/ConstAcceleration/Helper.hpp"
#include "Trajectory/ConstAcceleration/PoseExtrapolator.hpp"
#include "Trajectory/ConstAcceleration/PoseInterpolator.hpp"
#include "Trajectory/ConstAcceleration/PriorFactor.hpp"
#include "Trajectory/ConstAcceleration/VelocityExtrapolator.hpp"
#include "Trajectory/ConstAcceleration/VelocityInterpolator.hpp"

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

            void Interface::add(const Time& time, const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                    const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink,
                    const slam::eval::Evaluable<AccelerationType>::Ptr& dw_0k_ink) {
                // Check for duplicate time using an accessor
                KnotMap::const_accessor acc;
                if (knot_map_.find(acc, time))
                    throw std::runtime_error("[Interface::add] adding knot at duplicated time");

                // Create the new knot
                const auto knot = std::make_shared<Variable>(time, T_k0, w_0k_ink, dw_0k_ink);

                // Insert the knot into the concurrent hash map
                if (!knot_map_.insert(std::make_pair(time, knot)))
                    throw std::runtime_error("[Interface::add] failed to insert knot into map");
            }

            // -----------------------------------------------------------------------------
            // get
            // -----------------------------------------------------------------------------

            Variable::ConstPtr Interface::get(const Time& time) const {
                KnotMap::const_accessor acc;
                if (!knot_map_.find(acc, time))
                    throw std::out_of_range("[Interface::get] no knot found at provided time");
                return acc->second;
            }

            // -----------------------------------------------------------------------------
            // getPoseInterpolator
            // -----------------------------------------------------------------------------

            auto Interface::getPoseInterpolator(const Time& time) const
                    -> slam::eval::Evaluable<PoseType>::ConstPtr {
                // Check that map is not empty
                if (knot_map_.empty()) throw std::runtime_error("[Interface::getPoseInterpolator] knot map is empty");

                // Try exact match first
                KnotMap::const_accessor acc;
                if (knot_map_.find(acc, time)) {
                    return acc->second->getPose();
                }

                // No exact match, find bounding knots
                Time t1, t2;
                Variable::ConstPtr knot1 = nullptr, knot2 = nullptr;

                // Iterate over the map to locate the interval
                for (KnotMap::const_iterator it = knot_map_.begin(); it != knot_map_.end(); ++it) {
                    const Time& knot_time = it->first;
                    const auto& knot = it->second;

                    if (knot_time <= time) {
                    t1 = knot_time;
                    knot1 = knot;
                    } else if (knot_time > time && !knot2) {
                    t2 = knot_time;
                    knot2 = knot;
                    break; // Found upper bound, no need to continue
                    }
                }

                // Handle edge cases
                if (!knot1 && !knot2) {
                    throw std::runtime_error("[Interface::getPoseInterpolator] knot map iteration failed unexpectedly");
                }

                if (!knot1) {
                    // Time is before the first knot
                    return getPoseExtrapolator_(time, knot2);
                }

                if (!knot2) {
                    // Time is after the last knot
                    return getPoseExtrapolator_(time, knot1);
                }

                // Check if time is within the interval
                if (time <= t1 || time >= t2) {
                    throw std::runtime_error(
                        "[Interface::getPoseInterpolator] Requested interpolation at an invalid time: " +
                        std::to_string(time.seconds()) + " not in (" +
                        std::to_string(t1.seconds()) + ", " +
                        std::to_string(t2.seconds()) + ")");
                }

                // Create interpolated evaluator
                return getPoseInterpolator_(time, knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // getVelocityInterpolator
            // -----------------------------------------------------------------------------

            auto Interface::getVelocityInterpolator(const Time& time) const
                    -> slam::eval::Evaluable<VelocityType>::ConstPtr {
                // Check that map is not empty
                if (knot_map_.empty()) throw std::runtime_error("[Interface::getVelocityInterpolator] knot map is empty");

                // Try exact match first
                KnotMap::const_accessor acc;
                if (knot_map_.find(acc, time)) {
                    return acc->second->getVelocity();
                }

                // No exact match, find bounding knots
                Time t1, t2;
                Variable::ConstPtr knot1 = nullptr, knot2 = nullptr;

                // Iterate over the map to locate the interval
                for (KnotMap::const_iterator it = knot_map_.begin(); it != knot_map_.end(); ++it) {
                    const Time& knot_time = it->first;
                    const auto& knot = it->second;

                    if (knot_time <= time) {
                    t1 = knot_time;
                    knot1 = knot;
                    } else if (knot_time > time && !knot2) {
                    t2 = knot_time;
                    knot2 = knot;
                    break; // Found upper bound, no need to continue
                    }
                }

                // Handle edge cases
                if (!knot1 && !knot2) {
                    throw std::runtime_error("[Interface::getVelocityInterpolator] knot map iteration failed unexpectedly");
                }

                if (!knot1) {
                    // Time is before the first knot
                    return getVelocityExtrapolator_(time, knot2);
                }

                if (!knot2) {
                    // Time is after the last knot
                    return getVelocityExtrapolator_(time, knot1);
                }

                // Check if time is within the interval
                if (time <= t1 || time >= t2) {
                    throw std::runtime_error(
                        "[Interface::getVelocityInterpolator] Requested interpolation at an invalid time: " +
                        std::to_string(time.seconds()) + " not in (" +
                        std::to_string(t1.seconds()) + ", " +
                        std::to_string(t2.seconds()) + ")");
                }

                // Create interpolated evaluator
                return getVelocityInterpolator_(time, knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // getAccelerationInterpolator
            // -----------------------------------------------------------------------------

            auto Interface::getAccelerationInterpolator(const Time& time) const
                    -> slam::eval::Evaluable<AccelerationType>::ConstPtr {
                // Check that map is not empty
                if (knot_map_.empty()) throw std::runtime_error("[Interface::getAccelerationInterpolator] knot map is empty");

                // Try exact match first
                KnotMap::const_accessor acc;
                if (knot_map_.find(acc, time)) {
                    return acc->second->getAcceleration();
                }

                // No exact match, find bounding knots
                Time t1, t2;
                Variable::ConstPtr knot1 = nullptr, knot2 = nullptr;

                // Iterate over the map to locate the interval
                for (KnotMap::const_iterator it = knot_map_.begin(); it != knot_map_.end(); ++it) {
                    const Time& knot_time = it->first;
                    const auto& knot = it->second;

                    if (knot_time <= time) {
                    t1 = knot_time;
                    knot1 = knot;
                    } else if (knot_time > time && !knot2) {
                    t2 = knot_time;
                    knot2 = knot;
                    break; // Found upper bound, no need to continue
                    }
                }

                // Handle edge cases
                if (!knot1 && !knot2) {
                    throw std::runtime_error("[Interface::getAccelerationInterpolator] knot map iteration failed unexpectedly");
                }

                if (!knot1) {
                    // Time is before the first knot
                    return getAccelerationExtrapolator_(time, knot2);
                }

                if (!knot2) {
                    // Time is after the last knot
                    return getAccelerationExtrapolator_(time, knot1);
                }

                // Check if time is within the interval
                if (time <= t1 || time >= t2) {
                    throw std::runtime_error(
                        "[Interface::getAccelerationInterpolator] Requested interpolation at an invalid time: " +
                        std::to_string(time.seconds()) + " not in (" +
                        std::to_string(t1.seconds()) + ", " +
                        std::to_string(t2.seconds()) + ")");
                }

                // Create interpolated evaluator
                return getAccelerationInterpolator_(time, knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // getCovariance
            // -----------------------------------------------------------------------------

            auto Interface::getCovariance(const slam::solver::Covariance& cov, const Time& time) const
                    -> CovType {
                // Check that map is not empty
                if (knot_map_.empty()) throw std::runtime_error("map is empty");

                // Try exact match first
                KnotMap::const_accessor acc;
                if (knot_map_.find(acc, time)) {
                    const auto& knot = acc->second;
                    const auto T_k0 = knot->getPose();
                    const auto w_0k_ink = knot->getVelocity();
                    const auto dw_0k_ink = knot->getAcceleration();
                    if (!T_k0->active() || !w_0k_ink->active() || !dw_0k_ink->active())
                    throw std::runtime_error("extrapolation from a locked knot not implemented.");

                    const auto T_k0_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(T_k0);
                    const auto w_0k_ink_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(w_0k_ink);
                    const auto dw_0k_ink_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(dw_0k_ink);
                    if (!T_k0_var || !w_0k_ink_var || !dw_0k_ink_var)
                    throw std::runtime_error("trajectory states are not variables.");

                    std::vector<slam::eval::StateVariableBase::ConstPtr> state_var{T_k0_var, w_0k_ink_var, dw_0k_ink_var};
                    return cov.query(state_var);
                }

                // No exact match, find bounding knots
                Time t1, t2;
                Variable::ConstPtr knot1 = nullptr, knot2 = nullptr;

                // Iterate over the map to locate the interval
                for (KnotMap::const_iterator it = knot_map_.begin(); it != knot_map_.end(); ++it) {
                    const Time& knot_time = it->first;
                    const auto& knot = it->second;

                    if (knot_time <= time) {
                    t1 = knot_time;
                    knot1 = knot;
                    } else if (knot_time > time && !knot2) {
                    t2 = knot_time;
                    knot2 = knot;
                    break; // Found upper bound, no need to continue
                    }
                }

                // Handle edge cases
                if (!knot1 && !knot2) {
                    throw std::runtime_error("knot map iteration failed unexpectedly");
                }

                if (!knot1) {
                    throw std::runtime_error("Requested covariance before first time.");
                }

                if (!knot2) {
                    // Extrapolate after last knot
                    const auto& endKnot = knot1;
                    const auto T_k0 = endKnot->getPose();
                    const auto w_0k_ink = endKnot->getVelocity();
                    const auto dw_0k_ink = endKnot->getAcceleration();
                    if (!T_k0->active() || !w_0k_ink->active() || !dw_0k_ink->active())
                    throw std::runtime_error("extrapolation from a locked knot not implemented.");

                    const auto T_k0_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(T_k0);
                    const auto w_0k_ink_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(w_0k_ink);
                    const auto dw_0k_ink_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(dw_0k_ink);
                    if (!T_k0_var || !w_0k_ink_var || !dw_0k_ink_var)
                    throw std::runtime_error("trajectory states are not variables.");

                    // Construct a knot for the extrapolated state
                    const auto T_t_0 = getPoseExtrapolator_(time, endKnot);
                    const auto w_t_0 = getVelocityExtrapolator_(time, endKnot);
                    const auto dw_t_0 = getAccelerationExtrapolator_(time, endKnot);
                    const auto extrap_knot = Variable::MakeShared(time, T_t_0, w_t_0, dw_t_0);

                    // Compute Jacobians
                    const auto F_t1 = -getJacKnot1_(endKnot, extrap_knot);
                    const auto E_t1_inv = getJacKnot2_(endKnot, extrap_knot).inverse();

                    // Prior covariance
                    const auto Qt1 = getQ_((extrap_knot->getTime() - endKnot->getTime()).seconds(), Qc_diag_);

                    // End knot covariance
                    const std::vector<slam::eval::StateVariableBase::ConstPtr> state_var{T_k0_var, w_0k_ink_var, dw_0k_ink_var};
                    const Eigen::Matrix<double, 18, 18> P_end = cov.query(state_var);

                    // Compute covariance
                    return E_t1_inv * (F_t1 * P_end * F_t1.transpose() + Qt1) * E_t1_inv.transpose();
                }

                // Interpolation between knot1 and knot2
                const auto T_10 = knot1->getPose();
                const auto w_01_in1 = knot1->getVelocity();
                const auto dw_01_in1 = knot1->getAcceleration();
                const auto T_20 = knot2->getPose();
                const auto w_02_in2 = knot2->getVelocity();
                const auto dw_02_in2 = knot2->getAcceleration();
                if (!T_10->active() || !w_01_in1->active() || !dw_01_in1->active() ||
                    !T_20->active() || !w_02_in2->active() || !dw_02_in2->active())
                    throw std::runtime_error("extrapolation from a locked knot not implemented.");

                const auto T_10_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(T_10);
                const auto w_01_in1_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(w_01_in1);
                const auto dw_01_in1_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(dw_01_in1);
                const auto T_20_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(T_20);
                const auto w_02_in2_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(w_02_in2);
                const auto dw_02_in2_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(dw_02_in2);
                if (!T_10_var || !w_01_in1_var || !dw_01_in1_var || !T_20_var || !w_02_in2_var || !dw_02_in2_var)
                    throw std::runtime_error("trajectory states are not variables.");

                // Construct a knot for the interpolated state
                const auto T_q0_eval = getPoseInterpolator_(time, knot1, knot2);
                const auto w_0q_inq_eval = getVelocityInterpolator_(time, knot1, knot2);
                const auto dw_0q_inq_eval = getAccelerationInterpolator_(time, knot1, knot2);
                const auto knotq = Variable::MakeShared(time, T_q0_eval, w_0q_inq_eval, dw_0q_inq_eval);

                // Compute Jacobians
                const Eigen::Matrix<double, 18, 18> F_t1 = -getJacKnot1_(knot1, knotq);
                const Eigen::Matrix<double, 18, 18> E_t1 = getJacKnot2_(knot1, knotq);
                const Eigen::Matrix<double, 18, 18> F_2t = -getJacKnot1_(knotq, knot2);
                const Eigen::Matrix<double, 18, 18> E_2t = getJacKnot2_(knotq, knot2);

                // Prior inverse covariances
                const Eigen::Matrix<double, 18, 18> Qt1_inv = getQinv_((knotq->getTime() - knot1->getTime()).seconds(), Qc_diag_);
                const Eigen::Matrix<double, 18, 18> Q2t_inv = getQinv_((knot2->getTime() - knotq->getTime()).seconds(), Qc_diag_);

                // Covariance of knot1 and knot2
                const std::vector<slam::eval::StateVariableBase::ConstPtr> state_var{T_10_var, w_01_in1_var, dw_01_in1_var, T_20_var, w_02_in2_var, dw_02_in2_var};
                const Eigen::Matrix<double, 36, 36> P_1n2 = cov.query(state_var);

                // Helper matrices
                Eigen::Matrix<double, 36, 18> A = Eigen::Matrix<double, 36, 18>::Zero();
                A.block<18, 18>(0, 0) = F_t1.transpose() * Qt1_inv * E_t1;
                A.block<18, 18>(18, 0) = E_2t.transpose() * Q2t_inv * F_2t;

                Eigen::Matrix<double, 36, 36> B = Eigen::Matrix<double, 36, 36>::Zero();
                B.block<18, 18>(0, 0) = F_t1.transpose() * Qt1_inv * F_t1;
                B.block<18, 18>(18, 18) = E_2t.transpose() * Q2t_inv * E_2t;

                const Eigen::Matrix<double, 18, 18> F_21 = -getJacKnot1_(knot1, knot2);
                const Eigen::Matrix<double, 18, 18> E_21 = getJacKnot2_(knot1, knot2);
                const Eigen::Matrix<double, 18, 18> Q21_inv = getQinv_((knot2->getTime() - knot1->getTime()).seconds(), Qc_diag_);

                Eigen::Matrix<double, 36, 36> Pinv_comp = Eigen::Matrix<double, 36, 36>::Zero();
                Pinv_comp.block<18, 18>(0, 0) = F_21.transpose() * Q21_inv * F_21;
                Pinv_comp.block<18, 18>(18, 0) = -E_21.transpose() * Q21_inv * F_21;
                Pinv_comp.block<18, 18>(0, 18) = Pinv_comp.block<18, 18>(18, 0).transpose();
                Pinv_comp.block<18, 18>(18, 18) = E_21.transpose() * Q21_inv * E_21;

                // Interpolated covariance
                const Eigen::Matrix<double, 18, 18> P_t_inv = E_t1.transpose() * Qt1_inv * E_t1 + F_2t.transpose() * Q2t_inv * F_2t -
                                A.transpose() * (P_1n2.inverse() + B - Pinv_comp).inverse() * A;

                return P_t_inv.inverse();
            }

            // -----------------------------------------------------------------------------
            // addPosePrior
            // -----------------------------------------------------------------------------

            void Interface::addPosePrior(const Time& time, const PoseType& T_k0,
                             const Eigen::Matrix<double, 6, 6>& cov) {
                if (pose_prior_factor_ != nullptr)
                    throw std::runtime_error("[Interface::addPosePrior] can only add one pose prior.");

                // Check that map is not empty
                if (knot_map_.empty()) throw std::runtime_error("[Interface::addPosePrior] knot map is empty.");

                // Try to find knot at the specified time
                KnotMap::const_accessor acc;
                if (!knot_map_.find(acc, time))
                    throw std::runtime_error("[Interface::addPosePrior] no knot at provided time.");

                // Get reference to the knot
                const auto& knot = acc->second;

                // Check that the pose is not locked
                if (!knot->getPose()->active())
                    throw std::runtime_error("[Interface::addPosePrior] tried to add prior to locked pose.");

                // Set up loss function, noise model, and error function
                auto error_func = slam::eval::se3::se3_error(knot->getPose(), T_k0);
                auto noise_model = slam::problem::noisemodel::StaticNoiseModel<6>::MakeShared(cov);
                auto loss_func = slam::problem::lossfunc::L2LossFunc::MakeShared();

                // Create cost term
                pose_prior_factor_ = slam::problem::costterm::WeightedLeastSqCostTerm<6>::MakeShared(
                    error_func, noise_model, loss_func);
            }

            // -----------------------------------------------------------------------------
            // addVelocityPrior
            // -----------------------------------------------------------------------------

            void Interface::addVelocityPrior(const Time& time, const VelocityType& w_0k_ink,
                                 const Eigen::Matrix<double, 6, 6>& cov) {
                if (vel_prior_factor_ != nullptr)
                    throw std::runtime_error("[Interface::addVelocityPrior] can only add one velocity prior.");

                // Check that map is not empty
                if (knot_map_.empty()) throw std::runtime_error("[Interface::addVelocityPrior] knot map is empty.");

                // Try to find knot at the specified time
                KnotMap::const_accessor acc;
                if (!knot_map_.find(acc, time))
                    throw std::runtime_error("[Interface::addVelocityPrior] no knot at provided time.");

                // Get reference to the knot
                const auto& knot = acc->second;

                // Check that the velocity is not locked
                if (!knot->getVelocity()->active())
                    throw std::runtime_error("[Interface::addVelocityPrior] tried to add prior to locked velocity.");

                // Set up loss function, noise model, and error function
                auto error_func = slam::eval::vspace::vspace_error<6>(knot->getVelocity(), w_0k_ink);
                auto noise_model = slam::problem::noisemodel::StaticNoiseModel<6>::MakeShared(cov);
                auto loss_func = slam::problem::lossfunc::L2LossFunc::MakeShared();

                // Create cost term
                vel_prior_factor_ = slam::problem::costterm::WeightedLeastSqCostTerm<6>::MakeShared(
                    error_func, noise_model, loss_func);
            }

            // -----------------------------------------------------------------------------
            // addAccelerationPrior
            // -----------------------------------------------------------------------------

            void Interface::addAccelerationPrior(const Time& time,
                                     const AccelerationType& dw_0k_ink,
                                     const Eigen::Matrix<double, 6, 6>& cov) {
                // Early exit if prior exists (avoid unnecessary map access)
                if (acc_prior_factor_)  // nullptr check optimized with member directly
                    throw std::runtime_error("[Interface::addAccelerationPrior] can only add one acceleration prior.");

                // Check map emptiness once
                if (knot_map_.empty()) throw std::runtime_error("[Interface::addAccelerationPrior] knot map is empty.");

                // Find knot with minimal overhead
                KnotMap::const_accessor acc;
                if (!knot_map_.find(acc, time))
                    throw std::runtime_error("[Interface::addAccelerationPrior] no knot at provided time.");

                const auto& knot = acc->second;

                // Check acceleration state efficiently
                auto accel = knot->getAcceleration();
                if (!accel->active())
                    throw std::runtime_error("[Interface::addAccelerationPrior] tried to add prior to locked acceleration.");

                // Reuse static resources where possible
                static const auto loss_func = slam::problem::lossfunc::L2LossFunc::MakeShared();  // Singleton for L2 loss
                auto error_func = slam::eval::vspace::vspace_error<6>(accel, dw_0k_ink);
                auto noise_model = slam::problem::noisemodel::StaticNoiseModel<6>::MakeShared(cov);

                // Assign cost term directly
                acc_prior_factor_ = slam::problem::costterm::WeightedLeastSqCostTerm<6>::MakeShared(error_func, noise_model, loss_func);
            }

            // -----------------------------------------------------------------------------
            // addStatePrior
            // -----------------------------------------------------------------------------

            void Interface::addStatePrior(const Time& time, const PoseType& T_k0,
                              const VelocityType& w_0k_ink,
                              const AccelerationType& dw_0k_ink,
                              const CovType& cov) {
                // Only allow adding 1 prior
                if (pose_prior_factor_ || vel_prior_factor_ || acc_prior_factor_)
                    throw std::runtime_error("[Interface::addStatePrior] a pose/velocity/acceleration prior already exists.");

                if (state_prior_factor_)
                    throw std::runtime_error("[Interface::addStatePrior] can only add one state prior.");

                // Check that map is not empty
                if (knot_map_.empty()) throw std::runtime_error("[Interface::addStatePrior] knot map is empty.");

                // Try to find knot at the specified time
                KnotMap::const_accessor acc;
                if (!knot_map_.find(acc, time))
                    throw std::runtime_error("[Interface::addStatePrior] no knot at provided time.");

                // Get reference to the knot
                const auto& knot = acc->second;

                // Check that the pose, velocity, and acceleration are not locked
                if (!knot->getPose()->active() || !knot->getVelocity()->active() || !knot->getAcceleration()->active())
                    throw std::runtime_error("[Interface::addStatePrior] tried to add prior to locked state.");

                // Set up error functions, noise model, and loss function
                auto pose_error = slam::eval::se3::se3_error(knot->getPose(), T_k0);
                auto velo_error = slam::eval::vspace::vspace_error<6>(knot->getVelocity(), w_0k_ink);
                auto acc_error = slam::eval::vspace::vspace_error<6>(knot->getAcceleration(), dw_0k_ink);
                auto error_temp = slam::eval::vspace::merge<6, 6>(pose_error, velo_error);
                auto error_func = slam::eval::vspace::merge<12, 6>(error_temp, acc_error);
                auto noise_model = slam::problem::noisemodel::StaticNoiseModel<18>::MakeShared(cov);
                auto loss_func = slam::problem::lossfunc::L2LossFunc::MakeShared();

                // Create cost term
                state_prior_factor_ = slam::problem::costterm::WeightedLeastSqCostTerm<18>::MakeShared(
                    error_func, noise_model, loss_func);
            }

            // -----------------------------------------------------------------------------
            // addPriorCostTerms
            // -----------------------------------------------------------------------------

            void Interface::addPriorCostTerms(slam::problem::Problem& problem) const {
                // If empty, return none
                if (knot_map_.empty()) return;

                // Check for pose, velocity, or acceleration priors
                if (pose_prior_factor_) problem.addCostTerm(pose_prior_factor_);
                if (vel_prior_factor_) problem.addCostTerm(vel_prior_factor_);
                if (acc_prior_factor_) problem.addCostTerm(acc_prior_factor_);

                // All prior factors will use an L2 loss function
                const auto loss_function = std::make_shared<slam::problem::lossfunc::L2LossFunc>();

                // Collect knots into a sorted vector to process consecutive pairs
                std::vector<std::pair<Time, Variable::ConstPtr>> sorted_knots;
                auto range = knot_map_.range();
                for (auto it = range.begin(); it != range.end(); ++it) {
                    sorted_knots.emplace_back(it->first, it->second);
                }

                // Sort by time to ensure consecutive pairs
                std::sort(sorted_knots.begin(), sorted_knots.end(),
                            [](const auto& a, const auto& b) { return a.first < b.first; });

                // If fewer than 2 knots, no prior terms to add between knots
                if (sorted_knots.size() < 2) return;

                // Iterate through consecutive pairs of knots
                for (size_t i = 0; i < sorted_knots.size() - 1; ++i) {
                    const auto& knot1 = sorted_knots[i].second;
                    const auto& knot2 = sorted_knots[i + 1].second;

                    // Check if any of the variables are unlocked
                    if (knot1->getPose()->active() || knot1->getVelocity()->active() ||
                        knot1->getAcceleration()->active() || knot2->getPose()->active() ||
                        knot2->getVelocity()->active() || knot2->getAcceleration()->active()) {
                        // Generate information matrix for GP prior factor
                        auto Qinv = getQinv_((knot2->getTime() - knot1->getTime()).seconds(), Qc_diag_);
                        const auto noise_model =
                            std::make_shared<slam::problem::noisemodel::StaticNoiseModel<18>>(Qinv, slam::problem::noisemodel::NoiseType::INFORMATION);

                        // Create error function and cost term
                        const auto error_function = getPriorFactor_(knot1, knot2);
                        const auto cost_term = std::make_shared<slam::problem::costterm::WeightedLeastSqCostTerm<18>>(
                            error_function, noise_model, loss_function);

                        // Add cost term to the problem
                        problem.addCostTerm(cost_term);
                    }
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