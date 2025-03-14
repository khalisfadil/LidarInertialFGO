#include "Core/Trajectory/ConstVelocity/Interface.hpp"

#include "Core/Evaluable/se3/Evaluables.hpp"
#include "Core/Evaluable/vspace/Evaluables.hpp"
#include "Core/Problem/LossFunc/LossFunc.hpp"
#include "Core/Problem/NoiseModel/StaticNoiseModel.hpp"
#include "Core/Trajectory/ConstVelocity/Helper.hpp"
#include "Core/Trajectory/ConstVelocity/PoseExtrapolator.hpp"
#include "Core/Trajectory/ConstVelocity/PoseInterpolator.hpp"
#include "Core/Trajectory/ConstVelocity/PriorFactor.hpp"
#include "Core/Trajectory/ConstVelocity/VelocityInterpolator.hpp"

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

            void Interface::add(const Time& time, const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                    const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink) {
                // Check for duplicate time using an accessor
                KnotMap::const_accessor acc;
                if (knot_map_.find(acc, time))
                    throw std::runtime_error("adding knot at duplicated time.");

                // Create the new knot
                const auto knot = std::make_shared<Variable>(time, T_k0, w_0k_ink);

                // Insert the knot into the concurrent hash map
                if (!knot_map_.insert(std::make_pair(time, knot)))
                    throw std::runtime_error("failed to insert knot into map");
            }

            // -----------------------------------------------------------------------------
            // get() - Retrieves a state knot
            // -----------------------------------------------------------------------------

            Variable::ConstPtr Interface::get(const slam::traj::Time& time) const {
                typename KnotMap::const_accessor acc;
                if (knot_map_.find(acc, time)) {
                    return acc->second;
                }
                throw std::out_of_range("[Interface::get] No trajectory knot exists at time " + std::to_string(time.seconds()));
            }

            // -----------------------------------------------------------------------------
            // getPoseInterpolator() - Computes interpolated pose at given time
            // -----------------------------------------------------------------------------

            auto Interface::getPoseInterpolator(const Time& time) const
                    -> slam::eval::Evaluable<PoseType>::ConstPtr {
                // Check if map is empty
                if (knot_map_.empty()) throw std::runtime_error("[Interface::getPoseInterpolator] knot map is empty");

                // Try exact match first
                KnotMap::const_accessor acc;
                if (knot_map_.find(acc, time)) {
                    return acc->second->getPose();
                }

                // Use range-based iteration to find the bounding interval
                const auto range = knot_map_.range();
                Time t1, t2;
                Variable::ConstPtr knot1 = nullptr, knot2 = nullptr;

                for (auto it = range.begin(); it != range.end(); ++it) {
                    const Time& knot_time = it->first;
                    const auto& knot = it->second;

                    if (knot_time <= time) {
                    t1 = knot_time;
                    knot1 = knot;
                    } else if (knot_time > time) {
                    t2 = knot_time;
                    knot2 = knot;
                    break; // Found the upper bound, no need to continue
                    }
                }

                // Handle edge cases
                if (!knot1 && !knot2) {
                    throw std::runtime_error("[Interface::getPoseInterpolator] knot map iteration failed unexpectedly");
                }

                if (!knot1) {
                    // Time is before the first knot
                    return PoseExtrapolator::MakeShared(time, knot2);
                }

                if (!knot2) {
                    // Time is after the last knot
                    return PoseExtrapolator::MakeShared(time, knot1);
                }

                // Validate the interval
                if (time <= t1 || time >= t2) {
                    // Use a static string buffer to avoid repeated construction
                    static thread_local char buffer[128];
                    snprintf(buffer, sizeof(buffer),
                            "[Interface::getPoseInterpolator] Requested interpolation at invalid time: %.6f not in (%.6f, %.6f)",
                            time.seconds(), t1.seconds(), t2.seconds());
                    throw std::runtime_error(buffer);
                }

                // Return interpolated pose
                return PoseInterpolator::MakeShared(time, knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // getVelocityInterpolator() - Computes interpolated velocity at given time
            // -----------------------------------------------------------------------------

            auto Interface::getVelocityInterpolator(const Time& time) const
                    -> slam::eval::Evaluable<VelocityType>::ConstPtr {
                // Check that map is not empty
                if (knot_map_.empty()) throw std::runtime_error("[Interface::getVelocityInterpolator] knot map is empty");

                // Try exact match first
                KnotMap::const_accessor acc;
                if (knot_map_.find(acc, time)) {
                    // Exact match found
                    return acc->second->getVelocity();
                }

                // No exact match, find the bounding knots
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
                    break; // Found the upper bound, no need to continue
                    }
                }

                // Handle edge cases
                if (!knot1 && !knot2) {
                    throw std::runtime_error("[Interface::getVelocityInterpolator] knot map iteration failed unexpectedly");
                }

                if (!knot1) {
                    // Time is before the first knot
                    return knot2->getVelocity();
                }

                if (!knot2) {
                    // Time is after the last knot
                    return knot1->getVelocity();
                }

                // Check if time is within the interval
                if (time <= t1 || time >= t2) {
                    throw std::runtime_error(
                        "Requested interpolation at an invalid time: " +
                        std::to_string(time.seconds()) + " not in (" +
                        std::to_string(t1.seconds()) + ", " +
                        std::to_string(t2.seconds()) + ")");
                }

                // Create interpolated evaluator
                return VelocityInterpolator::MakeShared(time, knot1, knot2);
            }

            // -----------------------------------------------------------------------------
            // getCovariance() - Computes the propagated covariance at a given time
            // -----------------------------------------------------------------------------

            auto Interface::getCovariance(const slam::solver::Covariance& cov, const Time& time) const
                    -> CovType {
                // Check that map is not empty
                if (knot_map_.empty()) throw std::runtime_error("[Interface::getCovariance] map is empty");

                // Try exact match first
                KnotMap::const_accessor acc;
                if (knot_map_.find(acc, time)) {
                    const auto& knot = acc->second;
                    const auto T_k0 = knot->getPose();
                    const auto w_0k_ink = knot->getVelocity();
                    if (!T_k0->active() || !w_0k_ink->active())
                    throw std::runtime_error("[Interface::getCovariance] extrapolation from a locked knot not implemented.");

                    const auto T_k0_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(T_k0);
                    const auto w_0k_ink_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(w_0k_ink);
                    if (!T_k0_var || !w_0k_ink_var)
                    throw std::runtime_error("[Interface::getCovariance] trajectory states are not variables.");

                    std::vector<slam::eval::StateVariableBase::ConstPtr> state_var{T_k0_var, w_0k_ink_var};
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
                    throw std::runtime_error("[Interface::getCovariance] knot map iteration failed unexpectedly");
                }

                if (!knot1) {
                    // Extrapolate before first knot (not implemented in original, throwing error)
                    throw std::runtime_error("[Interface::getCovariance] Requested covariance before first time.");
                }

                if (!knot2) {
                    // Extrapolate after last knot
                    const auto& endKnot = knot1;
                    const auto T_k0 = endKnot->getPose();
                    const auto w_0k_ink = endKnot->getVelocity();
                    if (!T_k0->active() || !w_0k_ink->active())
                    throw std::runtime_error("[Interface::getCovariance] extrapolation from a locked knot not implemented.");

                    const auto T_k0_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(T_k0);
                    const auto w_0k_ink_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(w_0k_ink);
                    if (!T_k0_var || !w_0k_ink_var)
                    throw std::runtime_error("[Interface::getCovariance] trajectory states are not variables.");

                    // Construct a knot for the extrapolated state
                    const auto T_t_0 = PoseExtrapolator::MakeShared(time, endKnot);
                    const auto extrap_knot = Variable::MakeShared(time, T_t_0, endKnot->getVelocity());

                    // Compute Jacobians
                    const auto F_t1 = -getJacKnot1(endKnot, extrap_knot);
                    const auto E_t1_inv = getJacKnot3(endKnot, extrap_knot);

                    // Prior covariance
                    const auto Qt1 = getQ((extrap_knot->getTime() - endKnot->getTime()).seconds(), Qc_diag_);

                    // End knot covariance
                    const std::vector<slam::eval::StateVariableBase::ConstPtr> state_var{T_k0_var, w_0k_ink_var};
                    const Eigen::Matrix<double, 12, 12> P_end = cov.query(state_var);

                    // Compute covariance
                    return E_t1_inv * (F_t1 * P_end * F_t1.transpose() + Qt1) * E_t1_inv.transpose();
                }

                // Interpolation between knot1 and knot2
                const auto T_10 = knot1->getPose();
                const auto w_01_in1 = knot1->getVelocity();
                const auto T_20 = knot2->getPose();
                const auto w_02_in2 = knot2->getVelocity();
                if (!T_10->active() || !w_01_in1->active() || !T_20->active() || !w_02_in2->active())
                    throw std::runtime_error("[Interface::getCovariance] extrapolation from a locked knot not implemented.");

                const auto T_10_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(T_10);
                const auto w_01_in1_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(w_01_in1);
                const auto T_20_var = std::dynamic_pointer_cast<slam::eval::se3::SE3StateVariable>(T_20);
                const auto w_02_in2_var = std::dynamic_pointer_cast<slam::eval::vspace::VSpaceStateVar<6>>(w_02_in2);
                if (!T_10_var || !w_01_in1_var || !T_20_var || !w_02_in2_var)
                    throw std::runtime_error("[Interface::getCovariance] trajectory states are not variables.");

                // Construct a knot for the interpolated state
                const auto T_q0_eval = PoseInterpolator::MakeShared(time, knot1, knot2);
                const auto w_0q_inq_eval = VelocityInterpolator::MakeShared(time, knot1, knot2);
                const auto knotq = Variable::MakeShared(time, T_q0_eval, w_0q_inq_eval);

                // Compute Jacobians
                const Eigen::Matrix<double, 12, 12> F_t1 = -getJacKnot1(knot1, knotq);
                const Eigen::Matrix<double, 12, 12> E_t1 = getJacKnot2(knot1, knotq);
                const Eigen::Matrix<double, 12, 12> F_2t = -getJacKnot1(knotq, knot2);
                const Eigen::Matrix<double, 12, 12> E_2t = getJacKnot2(knotq, knot2);

                // Prior inverse covariances
                const Eigen::Matrix<double, 12, 12> Qt1_inv = getQinv((knotq->getTime() - knot1->getTime()).seconds(), Qc_diag_);
                const Eigen::Matrix<double, 12, 12> Q2t_inv = getQinv((knot2->getTime() - knotq->getTime()).seconds(), Qc_diag_);

                // Covariance of knot1 and knot2
                const std::vector<slam::eval::StateVariableBase::ConstPtr> state_var{T_10_var, w_01_in1_var, T_20_var, w_02_in2_var};
                const Eigen::Matrix<double, 24, 24> P_1n2 = cov.query(state_var);

                // Helper matrices
                Eigen::Matrix<double, 24, 12> A = Eigen::Matrix<double, 24, 12>::Zero();
                A.block<12, 12>(0, 0) = F_t1.transpose() * Qt1_inv * E_t1;
                A.block<12, 12>(12, 0) = E_2t.transpose() * Q2t_inv * F_2t;

                Eigen::Matrix<double, 24, 24> B = Eigen::Matrix<double, 24, 24>::Zero();
                B.block<12, 12>(0, 0) = F_t1.transpose() * Qt1_inv * F_t1;
                B.block<12, 12>(12, 12) = E_2t.transpose() * Q2t_inv * E_2t;

                const Eigen::Matrix<double, 12, 12> F_21 = -getJacKnot1(knot1, knot2);
                const Eigen::Matrix<double, 12, 12> E_21 = getJacKnot2(knot1, knot2);
                const Eigen::Matrix<double, 12, 12> Q21_inv = getQinv((knot2->getTime() - knot1->getTime()).seconds(), Qc_diag_);

                Eigen::Matrix<double, 24, 24> Pinv_comp = Eigen::Matrix<double, 24, 24>::Zero();
                Pinv_comp.block<12, 12>(0, 0) = F_21.transpose() * Q21_inv * F_21;
                Pinv_comp.block<12, 12>(12, 0) = -E_21.transpose() * Q21_inv * F_21;
                Pinv_comp.block<12, 12>(0, 12) = Pinv_comp.block<12, 12>(12, 0).transpose();
                Pinv_comp.block<12, 12>(12, 12) = E_21.transpose() * Q21_inv * E_21;

                // Interpolated covariance
                const Eigen::Matrix<double, 12, 12> P_t_inv = E_t1.transpose() * Qt1_inv * E_t1 + F_2t.transpose() * Q2t_inv * F_2t -
                                A.transpose() * (P_1n2.inverse() + B - Pinv_comp).inverse() * A;

                Eigen::Matrix<double, 12, 12> P_tau = P_t_inv.inverse();
                const Eigen::VectorXcd evalues = P_tau.eigenvalues();
                bool psd = true;
                for (uint i = 0; i < 12; ++i) {
                    if (evalues(i).real() < 0.0) {
                    psd = false;
                    break;
                    }
                }
                if (psd)
                    return P_tau;

                // Fix non-PSD matrix using symmetric PSD projection
                Eigen::Matrix<double, 12, 12> P_fix = 0.5 * (P_tau + P_tau.transpose()); // Symmetric part
                Eigen::EigenSolver<Eigen::Matrix<double, 12, 12>> es(P_fix);
                Eigen::MatrixXcd evec = es.eigenvectors();
                Eigen::VectorXcd eval = es.eigenvalues();
                for (uint i = 0; i < 12; ++i) {
                    if (eval(i).real() < 0.0) eval(i) = std::complex<double>(0.0, 0.0); // Zero out negative eigenvalues
                }
                return evec.real() * eval.real().asDiagonal() * evec.real().transpose();
            }

            // -----------------------------------------------------------------------------
            // addPosePrior
            // -----------------------------------------------------------------------------

            void Interface::addPosePrior(const Time& time, const PoseType& T_k0,
                             const Eigen::Matrix<double, 6, 6>& cov) {
                if (state_prior_factor_ != nullptr)
                    throw std::runtime_error("[Interface::addPosePrior] a state prior already exists.");

                if (pose_prior_factor_ != nullptr)
                    throw std::runtime_error("[Interface::addPosePrior] can only add one pose prior.");

                // Check that map is not empty
                if (knot_map_.empty()) throw std::runtime_error("[Interface::addPosePrior] knot map is empty.");

                // Try to find knot at the specified time
                KnotMap::accessor acc; // Use accessor since we might modify the map in a concurrent context
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
                // Only allow adding 1 prior
                if (state_prior_factor_ != nullptr)
                    throw std::runtime_error("[Interface::addVelocityPrior] a state prior already exists.");

                if (vel_prior_factor_ != nullptr)
                    throw std::runtime_error("[Interface::addVelocityPrior] can only add one velocity prior.");

                // Check that map is not empty
                if (knot_map_.empty()) throw std::runtime_error("[Interface::addVelocityPrior] knot map is empty.");

                // Try to find knot at the specified time
                KnotMap::accessor acc; // Use accessor for potential concurrent modification
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
            // addStatePrior
            // -----------------------------------------------------------------------------

            void Interface::addStatePrior(const Time& time, const PoseType& T_k0,
                              const VelocityType& w_0k_ink,
                              const CovType& cov) {
                // Only allow adding 1 prior
                if ((pose_prior_factor_ != nullptr) || (vel_prior_factor_ != nullptr))
                    throw std::runtime_error("[Interface::addStatePrior] a state prior already exists.");

                if (state_prior_factor_ != nullptr)
                    throw std::runtime_error("[Interface::addStatePrior] can only add one state prior.");

                // Check that map is not empty
                if (knot_map_.empty()) throw std::runtime_error("[Interface::addStatePrior] knot map is empty.");

                // Try to find knot at the specified time
                KnotMap::accessor acc; // Use accessor for potential concurrent modification
                if (!knot_map_.find(acc, time))
                    throw std::runtime_error("[Interface::addStatePrior] no knot at provided time.");

                // Get reference to the knot
                const auto& knot = acc->second;

                // Check that the pose and velocity are not locked
                if ((!knot->getPose()->active()) || (!knot->getVelocity()->active()))
                    throw std::runtime_error("[Interface::addStatePrior] tried to add prior to locked state.");

                // Set up error functions, noise model, and loss function
                auto pose_error = slam::eval::se3::se3_error(knot->getPose(), T_k0);
                auto velo_error = slam::eval::vspace::vspace_error<6>(knot->getVelocity(), w_0k_ink);
                auto error_func = slam::eval::vspace::merge<6, 6>(pose_error, velo_error);
                auto noise_model = slam::problem::noisemodel::StaticNoiseModel<12>::MakeShared(cov);
                auto loss_func = slam::problem::lossfunc::L2LossFunc::MakeShared();

                // Create cost term
                state_prior_factor_ = slam::problem::costterm::WeightedLeastSqCostTerm<12>::MakeShared(
                    error_func, noise_model, loss_func);
            }

            // -----------------------------------------------------------------------------
            // addPriorCostTerms() - Adds all prior constraints to the optimization problem
            // -----------------------------------------------------------------------------

            void Interface::addPriorCostTerms(slam::problem::Problem& problem) const {
                // If empty, return none
                if (knot_map_.empty()) return;

                // Check for pose or velocity priors
                if (pose_prior_factor_ != nullptr) problem.addCostTerm(pose_prior_factor_);
                if (vel_prior_factor_ != nullptr) problem.addCostTerm(vel_prior_factor_);
                if (state_prior_factor_ != nullptr) problem.addCostTerm(state_prior_factor_);

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
                        knot2->getPose()->active() || knot2->getVelocity()->active()) {
                        // Generate 12 x 12 information matrix for GP prior factor
                        auto Qinv = getQinv((knot2->getTime() - knot1->getTime()).seconds(), Qc_diag_);
                        const auto noise_model =
                            std::make_shared<slam::problem::noisemodel::StaticNoiseModel<12>>(Qinv, slam::problem::noisemodel::NoiseType::INFORMATION);

                        // Create error function and cost term
                        const auto error_function = PriorFactor::MakeShared(knot1, knot2);
                        const auto cost_term = std::make_shared<slam::problem::costterm::WeightedLeastSqCostTerm<12>>(
                            error_function, noise_model, loss_function);

                        // Add cost term to the problem
                        problem.addCostTerm(cost_term);
                    }
                }
            }
        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
