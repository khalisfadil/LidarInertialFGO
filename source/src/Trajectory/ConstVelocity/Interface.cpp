#include "source/include/Trajectory/ConstVelocity/Interface.hpp"

#include "source/include/Evaluable/se3/Evaluables.hpp"
#include "source/include/Evaluable/vspace/Evaluables.hpp"
#include "source/include/Problem/LossFunc/LossFunc.hpp"
#include "source/include/Problem/NoiseModel/StaticNoiseModel.hpp"
#include "source/include/Trajectory/ConstVelocity/Helper.hpp"
#include "source/include/Trajectory/ConstVelocity/PoseExtrapolator.hpp"
#include "source/include/Trajectory/ConstVelocity/PoseInterpolator.hpp"
#include "source/include/Trajectory/ConstVelocity/PriorFactor.hpp"
#include "source/include/Trajectory/ConstVelocity/VelocityInterpolator.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {

            // -----------------------------------------------------------------------------
            // Factory Method
            // -----------------------------------------------------------------------------

            auto Interface::MakeShared(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& Qc_diag) -> Ptr {
                return std::make_shared<Interface>(Qc_diag);
            }

            // -----------------------------------------------------------------------------
            // Constructor
            // -----------------------------------------------------------------------------

            Interface::Interface(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& Qc_diag)
                : Qc_diag_(Qc_diag) {}

            // -----------------------------------------------------------------------------
            // add() - Adds a new trajectory knot
            // -----------------------------------------------------------------------------

            void Interface::add(const slam::traj::Time& time,
                                const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                                const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink) {
                if (!T_k0 || !w_0k_ink) {
                    throw std::invalid_argument("[Interface::add] Pose or velocity evaluable is null.");
                }
                if (knot_map_.count(time)) {
                    throw std::runtime_error("[Interface::add] Duplicate trajectory knot at time " +
                                             std::to_string(time.seconds()));
                }
                knot_map_.emplace(time, std::make_shared<Variable>(time, T_k0, w_0k_ink));
            }

            // -----------------------------------------------------------------------------
            // get() - Retrieves a state knot
            // -----------------------------------------------------------------------------

            Variable::ConstPtr Interface::get(const slam::traj::Time& time) const {
                if (auto it = knot_map_.find(time); it != knot_map_.end()) {
                    return it->second;
                }
                throw std::out_of_range("[Interface::get] No trajectory knot exists at time " + std::to_string(time.seconds()));
            }

            // -----------------------------------------------------------------------------
            // getPoseInterpolator() - Computes interpolated pose at given time
            // -----------------------------------------------------------------------------

            auto Interface::getPoseInterpolator(const Time& time) const -> slam::eval::Evaluable<PoseType>::ConstPtr {
                if (knot_map_.empty()) 
                    throw std::runtime_error("[Interface::getPoseInterpolator] Knot map is empty");

                auto it_upper = knot_map_.lower_bound(time);

                // If `time` is beyond the last entry, extrapolate from the last knot
                if (it_upper == knot_map_.end()) 
                    return PoseExtrapolator::MakeShared(time, std::prev(it_upper)->second);

                // If `time` matches exactly, return the existing pose evaluator
                if (it_upper->second->getTime() == time) 
                    return it_upper->second->getPose();

                // If `time` is before the first entry, extrapolate from the first knot
                if (it_upper == knot_map_.begin()) 
                    return PoseExtrapolator::MakeShared(time, it_upper->second);

                // Get iterators bounding the interpolation interval
                auto it_lower = std::prev(it_upper);

                if (time <= it_lower->second->getTime() || time >= it_upper->second->getTime()) 
                    throw std::runtime_error("[Interface::getPoseInterpolator] Requested interpolation at an invalid time: " +
                                            std::to_string(time.seconds()) + " not in (" +
                                            std::to_string(it_lower->second->getTime().seconds()) + ", " +
                                            std::to_string(it_upper->second->getTime().seconds()) + ")");

                return PoseInterpolator::MakeShared(time, it_lower->second, it_upper->second);
            }

            // -----------------------------------------------------------------------------
            // getVelocityInterpolator() - Computes interpolated velocity at given time
            // -----------------------------------------------------------------------------

            auto Interface::getVelocityInterpolator(const Time& time) const -> slam::eval::Evaluable<VelocityType>::ConstPtr {
                if (knot_map_.empty()) 
                    throw std::runtime_error("[Interface::getVelocityInterpolator] Knot map is empty");

                auto it_upper = knot_map_.lower_bound(time);

                // If `time` is beyond the last entry, return velocity of the last knot
                if (it_upper == knot_map_.end()) 
                    return std::prev(it_upper)->second->getVelocity();

                // If `time` matches exactly, return the existing velocity evaluator
                if (it_upper->second->getTime() == time) 
                    return it_upper->second->getVelocity();

                // If `time` is before the first entry, return velocity of the first knot
                if (it_upper == knot_map_.begin()) 
                    return it_upper->second->getVelocity();

                // Get iterators bounding the interpolation interval
                auto it_lower = std::prev(it_upper);

                if (time <= it_lower->second->getTime() || time >= it_upper->second->getTime()) 
                    throw std::runtime_error("[Interface::getVelocityInterpolator] Requested interpolation at an invalid time: " +
                                            std::to_string(time.seconds()) + " not in (" +
                                            std::to_string(it_lower->second->getTime().seconds()) + ", " +
                                            std::to_string(it_upper->second->getTime().seconds()) + ")");

                return VelocityInterpolator::MakeShared(time, it_lower->second, it_upper->second);
            }

            // -----------------------------------------------------------------------------
            // getCovariance() - Computes the propagated covariance at a given time
            // -----------------------------------------------------------------------------

            auto Interface::getCovariance(const slam::solver::Covariance& cov, const Time& time) -> CovType {
                if (knot_map_.empty()) 
                    throw std::runtime_error("[Interface::getCovariance] Knot map is empty");

                auto it_upper = knot_map_.lower_bound(time);

                // Extrapolate after last entry
                if (it_upper == knot_map_.end()) {
                    const auto& endKnot = std::prev(it_upper)->second;

                    if (!endKnot->getPose()->active() || !endKnot->getVelocity()->active())
                        throw std::runtime_error("[Interface::getCovariance] Extrapolation from a locked knot not implemented.");

                    auto T_k0_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(endKnot->getPose());
                    auto w_0k_ink_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(endKnot->getVelocity());

                    if (!T_k0_var || !w_0k_ink_var)
                        throw std::runtime_error("[Interface::getCovariance] Trajectory states are not variables.");

                    auto extrap_knot = Variable::MakeShared(time, PoseExtrapolator::MakeShared(time, endKnot), endKnot->getVelocity());

                    auto F_t1 = -getJacKnot1(endKnot, extrap_knot);
                    auto E_t1_inv = getJacKnot3(endKnot, extrap_knot);
                    auto Qt1 = getQ((extrap_knot->getTime() - endKnot->getTime()).seconds(), Qc_diag_);
                    auto P_end = cov.query({T_k0_var, w_0k_ink_var});

                    return E_t1_inv * (F_t1 * P_end * F_t1.transpose() + Qt1) * E_t1_inv.transpose();
                }

                // If `time` matches exactly, return stored covariance
                if (it_upper->second->getTime() == time) {
                    const auto& knot = it_upper->second;

                    if (!knot->getPose()->active() || !knot->getVelocity()->active())
                        throw std::runtime_error("[Interface::getCovariance] Extrapolation from a locked knot not implemented.");

                    auto T_k0_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(knot->getPose());
                    auto w_0k_ink_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(knot->getVelocity());

                    if (!T_k0_var || !w_0k_ink_var)
                        throw std::runtime_error("[Interface::getCovariance] Trajectory states are not variables.");

                    return cov.query({T_k0_var, w_0k_ink_var});
                }

                if (it_upper == knot_map_.begin())
                    throw std::runtime_error("[Interface::getCovariance] Requested covariance before first time.");

                // Get iterators bounding the interpolation interval
                auto it_lower = std::prev(it_upper);
                const auto& knot1 = it_lower->second;
                const auto& knot2 = it_upper->second;

                if (!knot1->getPose()->active() || !knot1->getVelocity()->active() || 
                    !knot2->getPose()->active() || !knot2->getVelocity()->active()) {
                    throw std::runtime_error("[Interface::getCovariance] Extrapolation from a locked knot not implemented.");
                }

                auto T_10_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(knot1->getPose());
                auto w_01_in1_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(knot1->getVelocity());
                auto T_20_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(knot2->getPose());
                auto w_02_in2_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(knot2->getVelocity());

                if (!T_10_var || !w_01_in1_var || !T_20_var || !w_02_in2_var)
                    throw std::runtime_error("[Interface::getCovariance] Trajectory states are not variables.");

                auto knotq = Variable::MakeShared(time, 
                                                PoseInterpolator::MakeShared(time, knot1, knot2),
                                                VelocityInterpolator::MakeShared(time, knot1, knot2));

                auto F_t1 = -getJacKnot1(knot1, knotq);
                auto E_t1 = getJacKnot2(knot1, knotq);
                auto F_2t = -getJacKnot1(knotq, knot2);
                auto E_2t = getJacKnot2(knotq, knot2);

                auto Qt1_inv = getQinv((knotq->getTime() - knot1->getTime()).seconds(), Qc_diag_);
                auto Q2t_inv = getQinv((knot2->getTime() - knotq->getTime()).seconds(), Qc_diag_);

                auto P_1n2 = cov.query({T_10_var, w_01_in1_var, T_20_var, w_02_in2_var});

                Eigen::Matrix<double, 24, 12> A;
                A << F_t1.transpose() * Qt1_inv * E_t1, 
                    E_2t.transpose() * Q2t_inv * F_2t;

                Eigen::Matrix<double, 24, 24> B = Eigen::Matrix<double, 24, 24>::Zero();
                B.block<12, 12>(0, 0) = F_t1.transpose() * Qt1_inv * F_t1;
                B.block<12, 12>(12, 12) = E_2t.transpose() * Q2t_inv * E_2t;

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
                if (state_prior_factor_) 
                    throw std::runtime_error("[Interface::addPosePrior] A state prior already exists.");

                if (pose_prior_factor_) 
                    throw std::runtime_error("[Interface::addPosePrior] Can only add one pose prior.");

                if (knot_map_.empty()) 
                    throw std::runtime_error("[Interface::addPosePrior] Knot map is empty.");

                auto it = knot_map_.find(time);
                if (it == knot_map_.end()) 
                    throw std::runtime_error("[Interface::addPosePrior] No knot found at provided time.");

                const auto& knot = it->second;

                if (!knot->getPose()->active()) 
                    throw std::runtime_error("[Interface::addPosePrior] Attempted to add prior to a locked pose.");

                // Directly initialize cost term with error function, noise model, and loss function
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
                if (state_prior_factor_) 
                    throw std::runtime_error("[Interface::addVelocityPrior] A state prior already exists.");

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
            // addStatePrior
            // -----------------------------------------------------------------------------

            void Interface::addStatePrior(const Time& time, const PoseType& T_k0,
                              const VelocityType& w_0k_ink,
                              const CovType& cov) {
                if (state_prior_factor_)
                    throw std::runtime_error("[Interface::addStatePrior] Can only add one state prior.");

                if (pose_prior_factor_ || vel_prior_factor_)
                    throw std::runtime_error("[Interface::addStatePrior] A pose or velocity prior already exists.");

                if (knot_map_.empty())
                    throw std::runtime_error("[Interface::addStatePrior] Knot map is empty.");

                auto it = knot_map_.find(time);
                if (it == knot_map_.end())
                    throw std::runtime_error("[Interface::addStatePrior] No knot found at provided time.");

                const auto& knot = it->second;

                if (!knot->getPose()->active() || !knot->getVelocity()->active())
                    throw std::runtime_error("[Interface::addStatePrior] Attempted to add prior to a locked state.");

                // Directly initialize cost term with error function, noise model, and loss function
                state_prior_factor_ = slam::problem::costterm::WeightedLeastSqCostTerm<12>::MakeShared(
                    slam::eval::vspace::merge<6, 6>(
                        slam::eval::se3::se3_error(knot->getPose(), T_k0),
                        slam::eval::vspace::vspace_error<6>(knot->getVelocity(), w_0k_ink)),
                    slam::problem::noisemodel::StaticNoiseModel<12>::MakeShared(cov),
                    slam::problem::lossfunc::L2LossFunc::MakeShared());
            }

            // -----------------------------------------------------------------------------
            // addPriorCostTerms() - Adds all prior constraints to the optimization problem
            // -----------------------------------------------------------------------------

            void Interface::addPriorCostTerms(slam::problem::Problem& problem) const {
                // Return early if there are no knots
                if (knot_map_.empty()) return;

                // Add available prior factors efficiently
                if (pose_prior_factor_) problem.addCostTerm(pose_prior_factor_);
                if (vel_prior_factor_) problem.addCostTerm(vel_prior_factor_);
                if (state_prior_factor_) problem.addCostTerm(state_prior_factor_);

                // Use a shared L2 loss function for all cost terms
                static const auto loss_function = slam::problem::lossfunc::L2LossFunc::MakeShared();

                // Iterate over adjacent knots and add prior cost terms if any state is active
                for (auto it1 = knot_map_.begin(), it2 = std::next(it1); it2 != knot_map_.end(); ++it1, ++it2) {
                    const auto& knot1 = it1->second;
                    const auto& knot2 = it2->second;

                    // Skip if all states are locked
                    if (!(knot1->getPose()->active() || knot1->getVelocity()->active() ||
                        knot2->getPose()->active() || knot2->getVelocity()->active()))
                        continue;

                    // Compute information matrix for GP prior factor
                    const auto Qinv = getQinv((knot2->getTime() - knot1->getTime()).seconds(), Qc_diag_);
                    const auto noise_model = slam::problem::noisemodel::StaticNoiseModel<12>::MakeShared(Qinv, slam::problem::noisemodel::NoiseType::INFORMATION);

                    // Create and add cost term
                    problem.addCostTerm(slam::problem::costterm::WeightedLeastSqCostTerm<12>::MakeShared(
                        PriorFactor::MakeShared(knot1, knot2), noise_model, loss_function));
                }
            }
        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
