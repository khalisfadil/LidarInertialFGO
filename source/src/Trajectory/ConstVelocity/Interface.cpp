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

            auto Interface::MakeShared(const Eigen::Matrix<double, 6, 1>& Qc_diag) -> Ptr {
                return std::make_shared<Interface>(Qc_diag);
            }

            // -----------------------------------------------------------------------------
            // Constructor
            // -----------------------------------------------------------------------------

            Interface::Interface(const Eigen::Matrix<double, 6, 1>& Qc_diag)
                : Qc_diag_(Qc_diag) {}

            // -----------------------------------------------------------------------------
            // add() - Adds a new trajectory knot
            // -----------------------------------------------------------------------------

            void Interface::add(const slam::traj::Time& time,
                                const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                                const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink) {
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
                auto it = knot_map_.find(time);
                if (it == knot_map_.end()) {
                    throw std::runtime_error("[Interface::get] Requested state knot does not exist.");
                }
                return it->second;
            }

            // -----------------------------------------------------------------------------
            // getPoseInterpolator() - Computes interpolated pose at given time
            // -----------------------------------------------------------------------------

            auto Interface::getPoseInterpolator(const slam::traj::Time& time) const
                -> slam::eval::Evaluable<PoseType>::ConstPtr {
                
                if (knot_map_.empty()) throw std::runtime_error("[Interface::getPoseInterpolator] Knot map is empty.");

                auto it1 = knot_map_.lower_bound(time);

                // Case 1: Time is after the last entry (extrapolation needed)
                if (it1 == knot_map_.end()) {
                    --it1;
                    return PoseExtrapolator::MakeShared(time, it1->second);
                }

                // Case 2: Exact match found
                if (it1->second->getTime() == time) return it1->second->getPose();

                // Case 3: Time is before the first entry
                if (it1 == knot_map_.begin()) {
                    return PoseExtrapolator::MakeShared(time, it1->second);
                }

                // Case 4: Interpolation between two knots
                auto it2 = it1--;
                return PoseInterpolator::MakeShared(time, it1->second, it2->second);
            }

            // -----------------------------------------------------------------------------
            // getVelocityInterpolator() - Computes interpolated velocity at given time
            // -----------------------------------------------------------------------------

            auto Interface::getVelocityInterpolator(const slam::traj::Time& time) const
                -> slam::eval::Evaluable<VelocityType>::ConstPtr {
                
                if (knot_map_.empty()) throw std::runtime_error("[Interface::getVelocityInterpolator] Knot map is empty.");

                auto it1 = knot_map_.lower_bound(time);

                // Case 1: Time is after the last entry
                if (it1 == knot_map_.end()) {
                    --it1;
                    return it1->second->getVelocity();
                }

                // Case 2: Exact match found
                if (it1->second->getTime() == time) return it1->second->getVelocity();

                // Case 3: Time is before the first entry
                if (it1 == knot_map_.begin()) {
                    return it1->second->getVelocity();
                }

                // Case 4: Interpolation between two knots
                auto it2 = it1--;
                return VelocityInterpolator::MakeShared(time, it1->second, it2->second);
            }

            // -----------------------------------------------------------------------------
            // getCovariance() - Computes the propagated covariance at a given time
            // -----------------------------------------------------------------------------

            Interface::CovType Interface::getCovariance(const slam::solver::Covariance& cov,
                                            const slam::traj::Time& time) {
                // Retrieve the trajectory knot at the given time
                auto knot = get(time);
                if (!knot) {
                    throw std::runtime_error("[Interface::getCovariance] No state found at the given time.");
                }

                // Extract pose and velocity evaluables
                auto pose_evaluable = knot->getPose();
                auto vel_evaluable = knot->getVelocity();

                if (!pose_evaluable || !vel_evaluable) {
                    throw std::runtime_error("[Interface::getCovariance] Pose or velocity evaluable is null.");
                }

                // Cast to StateVariableBase (ensuring these are valid state variables)
                auto pose_var = std::dynamic_pointer_cast<const slam::eval::StateVariableBase>(pose_evaluable);
                auto vel_var = std::dynamic_pointer_cast<const slam::eval::StateVariableBase>(vel_evaluable);

                if (!(pose_var && vel_var)) {
                    throw std::runtime_error("[Interface::getCovariance] Failed to cast pose or velocity to StateVariableBase.");
                }

                // Query and return covariance matrix
                return cov.query({pose_var, vel_var});
            }

            // -----------------------------------------------------------------------------
            // addPriorCostTerms() - Adds all prior constraints to the optimization problem
            // -----------------------------------------------------------------------------

            void Interface::addPriorCostTerms(slam::problem::Problem& problem) const {
                if (knot_map_.empty()) return;

                if (pose_prior_factor_ != nullptr) problem.addCostTerm(pose_prior_factor_);
                if (vel_prior_factor_ != nullptr) problem.addCostTerm(vel_prior_factor_);
                if (state_prior_factor_ != nullptr) problem.addCostTerm(state_prior_factor_);

                const auto loss_function = std::make_shared<slam::problem::lossfunc::L2LossFunc>();

                auto it1 = knot_map_.begin();
                auto it2 = it1;
                ++it2;

                for (; it2 != knot_map_.end(); ++it1, ++it2) {
                    const auto& knot1 = it1->second;
                    const auto& knot2 = it2->second;

                    if (knot1->getPose()->active() || knot1->getVelocity()->active() ||
                        knot2->getPose()->active() || knot2->getVelocity()->active()) {

                        auto Qinv = getQinv((knot2->getTime() - knot1->getTime()).seconds(), Qc_diag_);
                        const auto noise_model = std::make_shared<slam::problem::noisemodel::StaticNoiseModel<12>>(Qinv, slam::problem::noisemodel::NoiseType::INFORMATION);
                        const auto error_function = PriorFactor::MakeShared(knot1, knot2);

                        const auto cost_term = std::make_shared<slam::problem::costterm::WeightedLeastSqCostTerm<12>>(
                            error_function, noise_model, loss_function);

                        problem.addCostTerm(cost_term);
                    }
                }
            }

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
