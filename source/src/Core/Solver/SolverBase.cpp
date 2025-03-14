#include <iostream>

#include "Core/Solver/SolverBase.hpp"
#include "Core/Common/Timer.hpp"

namespace slam {
    namespace solver {
        
        // -----------------------------------------------------------------------------
        // SolverBase
        // -----------------------------------------------------------------------------

        SolverBase::SolverBase(slam::problem::Problem& problem, const Params& params)
            : problem_(problem), state_vector_(problem.getStateVector()), params_(params) {
            
            if (!state_vector_) {
                throw std::runtime_error("[SolverBase::SolverBase] State vector is null.");
            }

            // Use `clone()` instead of direct assignment
            state_vector_backup_ = std::make_shared<slam::problem::StateVector>(state_vector_->clone());

            curr_cost_ = prev_cost_ = problem_.cost();
        }

        // -----------------------------------------------------------------------------
        // optimize
        // -----------------------------------------------------------------------------

        void SolverBase::optimize() {
            slam::common::Timer timer;
            while (!solver_converged_) iterate();
            if (params_.verbose) {
                std::cout << "[SolverBase::optimize] Total Optimization Time: " << timer.milliseconds() << " ms" << std::endl;
            }
        }

        // -----------------------------------------------------------------------------
        // iterate
        // -----------------------------------------------------------------------------

        void SolverBase::iterate() {
            if (solver_converged_) {
                std::cout << "[SolverBase::iterate] Requested an iteration when solver has already converged. Ignoring.\n";
                return;
            }

            if (params_.verbose && curr_iteration_ == 0) {
                std::cout << "Begin Optimization\n------------------\n";
                std::cout << "Number of States: " << state_vector_->getNumberOfStates() << std::endl;
                std::cout << "Number of Cost Terms: " << problem_.getNumberOfCostTerms() << std::endl;
                std::cout << "Initial Cost: " << curr_cost_ << std::endl;
            }

            curr_iteration_++;
            prev_cost_ = curr_cost_;

            double grad_norm = 0.0;
            bool step_success = linearizeSolveAndUpdate(curr_cost_, grad_norm);

            if (!step_success && fabs(grad_norm) < 1e-6) {
                term_ = TERMINATE_CONVERGED_ZERO_GRADIENT;
                solver_converged_ = true;
            } else if (!step_success) {
                term_ = TERMINATE_STEP_UNSUCCESSFUL;
                solver_converged_ = true;
                throw UnsuccessfulStep("[SolverBase::iterate] Solver terminated due to an unsuccessful step.");
            } else if (curr_iteration_ >= params_.max_iterations) {
                term_ = TERMINATE_MAX_ITERATIONS;
                solver_converged_ = true;
            } else if (curr_cost_ <= params_.absolute_cost_threshold) {
                term_ = TERMINATE_CONVERGED_ABSOLUTE_ERROR;
                solver_converged_ = true;
            } else if (fabs(prev_cost_ - curr_cost_) <= params_.absolute_cost_change_threshold) {
                term_ = TERMINATE_CONVERGED_ABSOLUTE_CHANGE;
                solver_converged_ = true;
            } else if (fabs(prev_cost_ - curr_cost_) / prev_cost_ <= params_.relative_cost_change_threshold) {
                term_ = TERMINATE_CONVERGED_RELATIVE_CHANGE;
                solver_converged_ = true;
            }

            if (params_.verbose && solver_converged_) {
                std::cout << "[SolverBase::iterate] Termination Cause: " << term_ << std::endl;
            }
        }

        // -----------------------------------------------------------------------------
        // proposeUpdate
        // -----------------------------------------------------------------------------

        double SolverBase::proposeUpdate(const Eigen::VectorXd& perturbation) {
            if (pending_proposed_state_) {
                throw std::runtime_error("[SolverBase::proposeUpdate] Already a pending update.");
            }

            state_vector_backup_ = std::make_shared<slam::problem::StateVector>(state_vector_->clone());
            state_vector_->update(perturbation);
            pending_proposed_state_ = true;
            return problem_.cost();
        }

        // -----------------------------------------------------------------------------
        // acceptProposedState
        // -----------------------------------------------------------------------------

        void SolverBase::acceptProposedState() {
            if (!pending_proposed_state_) {
                throw std::runtime_error("[SolverBase::acceptProposedState] No update proposed.");
            }
            pending_proposed_state_ = false;
        }

        // -----------------------------------------------------------------------------
        // rejectProposedState
        // -----------------------------------------------------------------------------

        void SolverBase::rejectProposedState() {
            if (!pending_proposed_state_) {
                throw std::runtime_error("[SolverBase::rejectProposedState] No update proposed.");
            }

            // Use copyValues instead of assignment to restore state
            state_vector_->copyValues(*state_vector_backup_);

            pending_proposed_state_ = false;
        }

        // -----------------------------------------------------------------------------
        // operator
        // -----------------------------------------------------------------------------

        std::ostream& operator<<(std::ostream& out, const SolverBase::Termination& T) {
            static const char* messages[] = {
                "NOT YET TERMINATED", "STEP UNSUCCESSFUL", "MAX ITERATIONS",
                "CONVERGED ABSOLUTE ERROR", "CONVERGED ABSOLUTE CHANGE",
                "CONVERGED RELATIVE CHANGE", "CONVERGED ZERO GRADIENT",
                "COST INCREASED"
            };
            out << messages[T];
            return out;
        }

    }  // namespace solver
}  // namespace slam
