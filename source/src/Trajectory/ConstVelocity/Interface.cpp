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
                    -> slam::eval::Evaluable<slam::liemath::se3::Transformation>::ConstPtr {

                // Check that the knot map is not empty
                if (knot_map_.empty()) {
                    throw std::runtime_error("[Interface::getPoseInterpolator] Knot map is empty.");
                }

                // Get iterator to the first element with time equal to or greater than 'time'
                auto it1 = knot_map_.lower_bound(time);

                // Check if time is passed the last entry
                if (it1 == knot_map_.end()) {
                    --it1;  // Safe, as we checked that the map isn't empty
                    const auto& endKnot = it1->second;
                    return slam::traj::const_vel::PoseExtrapolator::MakeShared(time, endKnot);  // Extrapolate pose
                }

                // Check if we requested time exactly
                if (it1->second->getTime() == time) {
                    return it1->second->getPose();  // Return the pose at the exact time
                }

                // Check if we requested before the first time
                if (it1 == knot_map_.begin()) {
                    const auto& startKnot = it1->second;
                    return slam::traj::const_vel::PoseExtrapolator::MakeShared(time, startKnot);  // Extrapolate pose
                }

                // Get iterators bounding the time interval
                auto it2 = it1;
                --it1;
                if (time <= it1->second->getTime() || time >= it2->second->getTime()) {
                    throw std::runtime_error(
                        "[Interface::getPoseInterpolator] Requested interpolation at an invalid time: " +
                        std::to_string(time.seconds()) + " not in (" +
                        std::to_string(it1->second->getTime().seconds()) + ", " +
                        std::to_string(it2->second->getTime().seconds()) + ")");
                }

                // Create interpolated evaluator for pose
                return slam::traj::const_vel::PoseInterpolator::MakeShared(time, it1->second, it2->second);
            }

            // -----------------------------------------------------------------------------
            // getVelocityInterpolator() - Computes interpolated velocity at given time
            // -----------------------------------------------------------------------------

            auto Interface::getVelocityInterpolator(const slam::traj::Time& time) const
                    -> slam::eval::Evaluable<Eigen::Matrix<double, 6, 1>>::ConstPtr {
                
                // Check that the knot map is not empty
                if (knot_map_.empty()) {
                    throw std::runtime_error("[Interface::getVelocityInterpolator] Knot map is empty.");
                }

                // Get iterator to the first element with time equal to or greater than 'time'
                auto it1 = knot_map_.lower_bound(time);

                // Check if time is passed the last entry
                if (it1 == knot_map_.end()) {
                    --it1;  // Safe, as we checked that the map is not empty
                    const auto& endKnot = it1->second;
                    return endKnot->getVelocity();  // Return velocity of the last knot
                }

                // Check if we requested time exactly
                if (it1->second->getTime() == time) {
                    return it1->second->getVelocity();  // Return velocity of the knot at the exact time
                }

                // Check if we requested before the first time
                if (it1 == knot_map_.begin()) {
                    const auto& startKnot = it1->second;
                    return startKnot->getVelocity();  // Return velocity of the first knot
                }

                // Get iterators bounding the time interval
                auto it2 = it1;
                --it1;
                if (time <= it1->second->getTime() || time >= it2->second->getTime()) {
                    throw std::runtime_error(
                        "[Interface::getVelocityInterpolator] Requested interpolation at an invalid time: " +
                        std::to_string(time.seconds()) + " not in (" +
                        std::to_string(it1->second->getTime().seconds()) + ", " +
                        std::to_string(it2->second->getTime().seconds()) + ")");
                }

                // Create interpolated evaluator for velocity
                return slam::traj::const_vel::VelocityInterpolator::MakeShared(time, it1->second, it2->second);
            }

            // -----------------------------------------------------------------------------
            // getCovariance() - Computes the propagated covariance at a given time
            // -----------------------------------------------------------------------------

            Interface::CovType Interface::getCovariance(const slam::solver::Covariance& cov,
                                            const slam::traj::Time& time) {
                if (knot_map_.empty()) {
                    throw std::runtime_error("[Interface::getCovariance] Knot map is empty.");
                }

                auto it1 = knot_map_.lower_bound(time);

                // Extrapolate after the last entry
                if (it1 == knot_map_.end()) {
                    --it1;  // Safe, as we've checked that the map isn't empty.
                    const auto& endKnot = it1->second;
                    const auto T_k0 = endKnot->getPose();
                    const auto w_0k_ink = endKnot->getVelocity();

                    if (!T_k0->active() || !w_0k_ink->active()) {
                        throw std::runtime_error("[Interface::getCovariance] Extrapolation from a locked knot not implemented.");
                    }

                    // Convert pose and velocity to state variables using SE3StateVariable and VSpaceStateVar
                    const auto T_k0_var = std::dynamic_pointer_cast<const slam::eval::se3::SE3StateVariable>(T_k0);
                    const auto w_0k_ink_var = std::dynamic_pointer_cast<const slam::eval::vspace::VSpaceStateVar<6>>(w_0k_ink);

                    if (!T_k0_var || !w_0k_ink_var) {
                        throw std::runtime_error("[Interface::getCovariance] Trajectory states are not variables.");
                    }

                    // Create an extrapolated state knot
                    const auto T_t_0 = PoseExtrapolator::MakeShared(time, endKnot);
                    const auto extrap_knot = Variable::MakeShared(time, T_t_0, endKnot->getVelocity());

                    // Compute Jacobians
                    const auto F_t1 = -getJacKnot1(endKnot, extrap_knot);
                    const auto E_t1_inv = getJacKnot3(endKnot, extrap_knot);

                    // Process noise covariance
                    const auto Qt1 = getQ((extrap_knot->getTime() - endKnot->getTime()).seconds(), Qc_diag_);

                    // Covariance at the end knot
                    std::vector<slam::eval::StateVariableBase::ConstPtr> state_vars = {T_k0_var, w_0k_ink_var};
                    const Eigen::Matrix<double, 12, 12> P_end = cov.query(state_vars, state_vars);

                    // Propagate covariance
                    return E_t1_inv * (F_t1 * P_end * F_t1.transpose() + Qt1) * E_t1_inv.transpose();
                }

                // Exact match: Return directly
                if (it1->second->getTime() == time) {
                    const auto& knot = it1->second;
                    const auto T_k0 = knot->getPose();
                    const auto w_0k_ink = knot->getVelocity();

                    if (!T_k0->active() || !w_0k_ink->active()) {
                        throw std::runtime_error("[Interface::getCovariance] Knot is locked.");
                    }

                    // Convert to state variables and query covariance
                    const auto T_k0_var = std::dynamic_pointer_cast<const slam::eval::se3::SE3StateVariable>(T_k0);
                    const auto w_0k_ink_var = std::dynamic_pointer_cast<const slam::eval::vspace::VSpaceStateVar<6>>(w_0k_ink);

                    if (!T_k0_var || !w_0k_ink_var) {
                        throw std::runtime_error("[Interface::getCovariance] Trajectory states are not variables.");
                    }

                    std::vector<slam::eval::StateVariableBase::ConstPtr> state_vars = {T_k0_var, w_0k_ink_var};
                    return cov.query(state_vars, state_vars);
                }

                // Covariance before the first knot is not supported
                if (it1 == knot_map_.begin()) {
                    throw std::runtime_error("[Interface::getCovariance] Requested covariance before first time.");
                }

                // Interpolation: Compute covariance between two knots
                auto it2 = it1--;
                const auto& knot1 = it1->second;
                const auto& knot2 = it2->second;

                // Convert pose and velocity to state variables using SE3StateVariable and VSpaceStateVar
                auto pose_var1 = std::dynamic_pointer_cast<const slam::eval::se3::SE3StateVariable>(knot1->getPose());
                auto vel_var1 = std::dynamic_pointer_cast<const slam::eval::vspace::VSpaceStateVar<6>>(knot1->getVelocity());
                auto pose_var2 = std::dynamic_pointer_cast<const slam::eval::se3::SE3StateVariable>(knot2->getPose());
                auto vel_var2 = std::dynamic_pointer_cast<const slam::eval::vspace::VSpaceStateVar<6>>(knot2->getVelocity());

                if (!pose_var1 || !vel_var1 || !pose_var2 || !vel_var2) {
                    throw std::runtime_error("[Interface::getCovariance] One or more trajectory states are not variables.");
                }

                // Create interpolated knot
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

                // Query covariance of knot1 and knot2
                std::vector<slam::eval::StateVariableBase::ConstPtr> state_vars = {pose_var1, vel_var1, pose_var2, vel_var2};
                const Eigen::Matrix<double, 24, 24> P_1n2 = cov.query(state_vars, state_vars);

                // Helper matrices for interpolation
                Eigen::Matrix<double, 24, 12> A = Eigen::Matrix<double, 24, 12>::Zero();
                A.block<12, 12>(0, 0) = F_t1.transpose() * Qt1_inv * E_t1;
                A.block<12, 12>(12, 0) = E_2t.transpose() * Q2t_inv * F_2t;

                Eigen::Matrix<double, 24, 24> B = Eigen::Matrix<double, 24, 24>::Zero();
                B.block<12, 12>(0, 0) = F_t1.transpose() * Qt1_inv * F_t1;
                B.block<12, 12>(12, 12) = E_2t.transpose() * Q2t_inv * E_2t;

                // Final interpolated covariance
                const Eigen::Matrix<double, 12, 12> P_t_inv = E_t1.transpose() * Qt1_inv * E_t1 +
                                                                        F_2t.transpose() * Q2t_inv * F_2t -
                                                                        A.transpose() * (P_1n2.inverse() + B).inverse() * A;

                Eigen::Matrix<double, 12, 12> P_tau = P_t_inv.inverse();

                // Ensure the matrix is positive semi-definite
                Eigen::EigenSolver<Eigen::Matrix<double, 12, 12>> es(0.5 * (P_tau + P_tau.transpose()));
                Eigen::MatrixXcd evec = es.eigenvectors();
                Eigen::VectorXcd eval = es.eigenvalues();

                for (uint i = 0; i < 12; ++i) {
                    if (eval(i).real() < 0.0) {
                        eval(i) = std::complex<double>(0.0, 0.0); // Set negative eigenvalues to zero
                    }
                }

                return evec.real() * eval.real().asDiagonal() * evec.real().transpose();
            }


            // -----------------------------------------------------------------------------
            // addPosePrior
            // -----------------------------------------------------------------------------

            void Interface::addPosePrior(const slam::traj::Time& time, 
                              const PoseType& T_k0,
                              const Eigen::Matrix<double, 6, 6>& cov) {
                // Only allow adding one prior
                if (state_prior_factor_ != nullptr) {
                    throw std::runtime_error("[Interface::addPosePrior] A state prior already exists.");
                }

                if (pose_prior_factor_ != nullptr) {
                    throw std::runtime_error("[Interface::addPosePrior] Can only add one pose prior.");
                }

                // Check that map is not empty
                if (knot_map_.empty()) {
                    throw std::runtime_error("[Interface::addPosePrior] Knot map is empty.");
                }

                // Try to find the knot at the provided time
                auto it = knot_map_.find(time);
                if (it == knot_map_.end()) {
                    throw std::runtime_error("[Interface::addPosePrior] No knot at provided time.");
                }

                // Get reference to the knot
                const auto& knot = it->second;

                // Check that the pose is not locked
                auto pose_var = std::dynamic_pointer_cast<const slam::eval::se3::SE3StateVariable>(knot->getPose());
                if (!pose_var || !pose_var->active()) {
                    throw std::runtime_error("[Interface::addPosePrior] Tried to add prior to a locked pose.");
                }

                // Create the SE(3) error evaluator for the pose prior
                auto error_func = slam::eval::se3::se3_error(knot->getPose(), T_k0);

                // Create the noise model
                auto noise_model = slam::problem::noisemodel::StaticNoiseModel<6>::MakeShared(cov);

                // Create the loss function (L2 Loss)
                auto loss_func = slam::problem::lossfunc::L2LossFunc::MakeShared();

                // Create the cost term for the pose prior
                pose_prior_factor_ = slam::problem::costterm::WeightedLeastSqCostTerm<6>::MakeShared(
                    error_func, noise_model, loss_func);
            }

            // -----------------------------------------------------------------------------
            // addVelocityPrior
            // -----------------------------------------------------------------------------

            void Interface::addVelocityPrior(const slam::traj::Time& time, 
                                 const VelocityType& w_0k_ink,
                                 const Eigen::Matrix<double, 6, 6>& cov) {
                // Only allow adding one prior
                if (state_prior_factor_ != nullptr) {
                    throw std::runtime_error("[Interface::addVelocityPrior] A state prior already exists.");
                }

                if (vel_prior_factor_ != nullptr) {
                    throw std::runtime_error("[Interface::addVelocityPrior] Can only add one velocity prior.");
                }

                // Check that map is not empty
                if (knot_map_.empty()) {
                    throw std::runtime_error("[Interface::addVelocityPrior] Knot map is empty.");
                }

                // Try to find the knot at the provided time
                auto it = knot_map_.find(time);
                if (it == knot_map_.end()) {
                    throw std::runtime_error("[Interface::addVelocityPrior] No knot at provided time.");
                }

                // Get reference to the knot
                const auto& knot = it->second;

                // Check that the velocity is not locked
                auto vel_var = std::dynamic_pointer_cast<const slam::eval::vspace::VSpaceStateVar<6>>(knot->getVelocity());
                if (!vel_var || !vel_var->active()) {
                    throw std::runtime_error("[Interface::addVelocityPrior] Tried to add prior to a locked velocity.");
                }

                // Create the velocity error evaluator for the velocity prior
                auto error_func = slam::eval::vspace::vspace_error<6>(knot->getVelocity(), w_0k_ink);

                // Create the noise model
                auto noise_model = slam::problem::noisemodel::StaticNoiseModel<6>::MakeShared(cov);

                // Create the loss function (L2 Loss)
                auto loss_func = slam::problem::lossfunc::L2LossFunc::MakeShared();

                // Create the cost term for the velocity prior
                vel_prior_factor_ = slam::problem::costterm::WeightedLeastSqCostTerm<6>::MakeShared(
                    error_func, noise_model, loss_func);
            }

            // -----------------------------------------------------------------------------
            // addStatePrior
            // -----------------------------------------------------------------------------

            void Interface::addStatePrior(const slam::traj::Time& time, 
                              const PoseType& T_k0, 
                              const VelocityType& w_0k_ink, 
                              const CovType& cov) {
                // Only allow adding one prior
                if ((pose_prior_factor_ != nullptr) || (vel_prior_factor_ != nullptr)) {
                    throw std::runtime_error("[Interface::addStatePrior] A pose/velocity prior already exists.");
                }

                if (state_prior_factor_ != nullptr) {
                    throw std::runtime_error("[Interface::addStatePrior] Can only add one state prior.");
                }

                // Check that the knot map is not empty
                if (knot_map_.empty()) {
                    throw std::runtime_error("[Interface::addStatePrior] Knot map is empty.");
                }

                // Try to find the knot at the provided time
                auto it = knot_map_.find(time);
                if (it == knot_map_.end()) {
                    throw std::runtime_error("[Interface::addStatePrior] No knot at provided time.");
                }

                // Get reference to the knot
                const auto& knot = it->second;

                // Check that the pose and velocity are not locked
                auto pose_var = std::dynamic_pointer_cast<const slam::eval::se3::SE3StateVariable>(knot->getPose());
                auto vel_var = std::dynamic_pointer_cast<const slam::eval::vspace::VSpaceStateVar<6>>(knot->getVelocity());
                if (!pose_var || !pose_var->active() || !vel_var || !vel_var->active()) {
                    throw std::runtime_error("[Interface::addStatePrior] Tried to add prior to a locked state.");
                }

                // Create the pose error evaluator
                auto pose_error = slam::eval::se3::se3_error(knot->getPose(), T_k0);
                
                // Create the velocity error evaluator
                auto velo_error = slam::eval::vspace::vspace_error<6>(knot->getVelocity(), w_0k_ink);
                
                // Combine the pose and velocity error functions
                auto error_func = slam::eval::vspace::merge<6, 6>(pose_error, velo_error);

                // Create the noise model
                auto noise_model = slam::problem::noisemodel::StaticNoiseModel<12>::MakeShared(cov);

                // Create the loss function (L2 Loss)
                auto loss_func = slam::problem::lossfunc::L2LossFunc::MakeShared();

                // Create the cost term for the state prior
                state_prior_factor_ = slam::problem::costterm::WeightedLeastSqCostTerm<12>::MakeShared(
                    error_func, noise_model, loss_func);
            }

            // -----------------------------------------------------------------------------
            // addPriorCostTerms() - Adds all prior constraints to the optimization problem
            // -----------------------------------------------------------------------------

            void Interface::addPriorCostTerms(slam::problem::Problem& problem) const {
                // If the knot map is empty, return
                if (knot_map_.empty()) return;

                // Check for pose, velocity, or state priors
                if (pose_prior_factor_ != nullptr) problem.addCostTerm(pose_prior_factor_);
                if (vel_prior_factor_ != nullptr) problem.addCostTerm(vel_prior_factor_);
                if (state_prior_factor_ != nullptr) problem.addCostTerm(state_prior_factor_);

                // All prior factors will use an L2 loss function
                const auto loss_function = std::make_shared<slam::problem::lossfunc::L2LossFunc>();

                // Initialize iterators for knot_map_
                auto it1 = knot_map_.begin();
                auto it2 = it1;
                ++it2;

                // Iterate through the knots to add prior terms
                for (; it2 != knot_map_.end(); ++it1, ++it2) {
                    // Get the knots
                    const auto& knot1 = it1->second;
                    const auto& knot2 = it2->second;

                    // Check if any of the pose or velocity variables are active
                    if (knot1->getPose()->active() || knot1->getVelocity()->active() ||
                        knot2->getPose()->active() || knot2->getVelocity()->active()) {

                        // Generate the 12x12 information matrix for the prior factor
                        auto Qinv = getQinv((knot2->getTime() - knot1->getTime()).seconds(), Qc_diag_);
                        const auto noise_model =
                            std::make_shared<slam::problem::noisemodel::StaticNoiseModel<12>>(Qinv, slam::problem::noisemodel::NoiseType::INFORMATION);

                        // Create the prior factor for constant velocity motion model
                        const auto prior_factor = slam::traj::const_vel::PriorFactor::MakeShared(knot1, knot2);

                        // Create the cost term for the prior
                        const auto cost_term = std::make_shared<slam::problem::costterm::WeightedLeastSqCostTerm<12>>(
                            prior_factor, noise_model, loss_function);

                        // Add the cost term to the problem
                        problem.addCostTerm(cost_term);
                    }
                }
            }
        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
