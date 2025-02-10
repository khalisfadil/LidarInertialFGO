#pragma once

#include <stdexcept>
#include <memory>
#include <Eigen/Core>

#include "source/include/Problem/Problem.hpp"
#include "source/include/Problem/StateVector.hpp"

namespace slam {
    namespace solver {

        // -----------------------------------------------------------------------------
        /** 
         * @brief Exception for solver failures (e.g. LLT decomposition failure).
         */
        class SolverFailure : public std::runtime_error {
        public:
            explicit SolverFailure(const std::string& message) : std::runtime_error(message) {}
        };

        // -----------------------------------------------------------------------------
        /** 
         * @brief Exception for unsuccessful steps that should be reported to the user.
         */
        class UnsuccessfulStep : public SolverFailure {
        public:
            explicit UnsuccessfulStep(const std::string& message) : SolverFailure(message) {}
        };

        // -----------------------------------------------------------------------------
        /**
         * @class SolverBase
         * @brief Base class for optimization solvers.
         */
        class SolverBase {
            public:
                struct Params {
                    virtual ~Params() = default;

                    /// Whether to print solver logs.
                    bool verbose = false;
                    /// Maximum number of iterations.
                    unsigned int max_iterations = 100;
                    /// Absolute cost threshold for convergence.
                    double absolute_cost_threshold = 0.0;
                    /// Absolute change in cost threshold for convergence.
                    double absolute_cost_change_threshold = 1e-4;
                    /// Relative change in cost threshold for convergence.
                    double relative_cost_change_threshold = 1e-4;
                };

                enum Termination {
                    TERMINATE_NOT_YET_TERMINATED,
                    TERMINATE_STEP_UNSUCCESSFUL,
                    TERMINATE_MAX_ITERATIONS,
                    TERMINATE_CONVERGED_ABSOLUTE_ERROR,
                    TERMINATE_CONVERGED_ABSOLUTE_CHANGE,
                    TERMINATE_CONVERGED_RELATIVE_CHANGE,
                    TERMINATE_CONVERGED_ZERO_GRADIENT,
                    TERMINATE_COST_INCREASED,
                    TERMINATE_EXPECTED_DELTA_COST_CONVERGED,
                };

                explicit SolverBase(slam::problem::Problem& problem, const Params& params);
                virtual ~SolverBase() = default;

                Termination terminationCause() const { return term_; }
                unsigned int currentIteration() const { return curr_iteration_; }

                void optimize();

            protected:
                void iterate();
                double proposeUpdate(const Eigen::VectorXd& perturbation);
                void acceptProposedState();
                void rejectProposedState();

                slam::problem::StateVector::ConstPtr stateVector() const { return state_vector_; }

                // -----------------------------------------------------------------------------
                /** @brief Reference to the optimization problem */
                slam::problem::Problem& problem_;

                // -----------------------------------------------------------------------------
                /** @brief Pointer to the state vector */
                slam::problem::StateVector::Ptr state_vector_;

                // -----------------------------------------------------------------------------
                /** @brief Backup state vector for reverting to previous values */
                slam::problem::StateVector::Ptr state_vector_backup_;

                Termination term_ = TERMINATE_NOT_YET_TERMINATED;
                unsigned int curr_iteration_ = 0;
                bool solver_converged_ = false;
                double curr_cost_ = 0.0;
                double prev_cost_ = 0.0;
                bool pending_proposed_state_ = false;

            private:
                virtual bool linearizeSolveAndUpdate(double& cost, double& grad_norm) = 0;
                const Params params_;
        };

        std::ostream& operator<<(std::ostream& out, const SolverBase::Termination& T);

    }  // namespace solver
}  // namespace slam
