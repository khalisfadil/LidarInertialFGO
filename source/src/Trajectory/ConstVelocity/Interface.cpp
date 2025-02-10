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
                
                if (knot_map_.empty()) {
                    throw std::runtime_error("[Interface::getPoseInterpolator] Knot map is empty.");
                }

                auto it1 = knot_map_.lower_bound(time);

                if (it1 == knot_map_.end()) {  // Extrapolation needed
                    return PoseExtrapolator::MakeShared(time, std::prev(it1)->second);
                }

                if (it1->second->getTime() == time) {  // Exact match
                    return it1->second->getPose();
                }

                if (it1 == knot_map_.begin()) {  // Before first entry
                    return PoseExtrapolator::MakeShared(time, it1->second);
                }

                return PoseInterpolator::MakeShared(time, std::prev(it1)->second, it1->second);
            }

            // -----------------------------------------------------------------------------
            // getVelocityInterpolator() - Computes interpolated velocity at given time
            // -----------------------------------------------------------------------------

            auto Interface::getVelocityInterpolator(const slam::traj::Time& time) const
                -> slam::eval::Evaluable<VelocityType>::ConstPtr {
                
                if (knot_map_.empty()) {
                    throw std::runtime_error("[Interface::getVelocityInterpolator] Knot map is empty.");
                }

                auto it1 = knot_map_.lower_bound(time);

                if (it1 == knot_map_.end()) {  // Extrapolation needed
                    return std::prev(it1)->second->getVelocity();
                }

                if (it1->second->getTime() == time) {  // Exact match
                    return it1->second->getVelocity();
                }

                if (it1 == knot_map_.begin()) {  // Before first entry
                    return it1->second->getVelocity();
                }

                return VelocityInterpolator::MakeShared(time, std::prev(it1)->second, it1->second);
            }

            // -----------------------------------------------------------------------------
            // getCovariance() - Computes the propagated covariance at a given time
            // -----------------------------------------------------------------------------

            Interface::CovType Interface::getCovariance(const slam::solver::Covariance& cov,
                                            const slam::traj::Time& time) {
                auto knot = get(time);
                if (!knot) {
                    throw std::runtime_error("[Interface::getCovariance] No state found at the given time.");
                }

                auto pose_var = std::dynamic_pointer_cast<const slam::eval::StateVariableBase>(knot->getPose());
                auto vel_var = std::dynamic_pointer_cast<const slam::eval::StateVariableBase>(knot->getVelocity());

                if (!pose_var || !vel_var) {
                    throw std::runtime_error("[Interface::getCovariance] Pose or velocity is not a state variable.");
                }

                return cov.query({pose_var, vel_var});
            }

            // -----------------------------------------------------------------------------
            // addPriorCostTerms() - Adds all prior constraints to the optimization problem
            // -----------------------------------------------------------------------------

            void Interface::addPriorCostTerms(slam::problem::Problem& problem) const {
                if (knot_map_.empty()) return;

                if (pose_prior_factor_) problem.addCostTerm(pose_prior_factor_);
                if (vel_prior_factor_) problem.addCostTerm(vel_prior_factor_);
                if (state_prior_factor_) problem.addCostTerm(state_prior_factor_);

                auto loss_function = std::make_shared<slam::problem::lossfunc::L2LossFunc>();

                for (auto it1 = knot_map_.begin(), it2 = std::next(it1); it2 != knot_map_.end(); ++it1, ++it2) {
                    const auto& knot1 = it1->second;
                    const auto& knot2 = it2->second;

                    if (!(knot1->getPose()->active() || knot1->getVelocity()->active() ||
                          knot2->getPose()->active() || knot2->getVelocity()->active())) {
                        continue;
                    }

                    auto Qinv = getQinv((knot2->getTime() - knot1->getTime()).seconds(), Qc_diag_);
                    auto noise_model = std::make_shared<slam::problem::noisemodel::StaticNoiseModel<12>>(
                        Qinv, slam::problem::noisemodel::NoiseType::INFORMATION);
                    auto error_function = PriorFactor::MakeShared(knot1, knot2);

                    auto cost_term = std::make_shared<slam::problem::costterm::WeightedLeastSqCostTerm<12>>(
                        error_function, noise_model, loss_function);

                    problem.addCostTerm(cost_term);
                }
            }

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
