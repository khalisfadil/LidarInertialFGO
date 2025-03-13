#pragma once

#include <stdexcept>
#include <memory>
#include <Eigen/Core>

#include "Problem/Problem.hpp"
#include "Problem/StateVector.hpp"

namespace slam {
    namespace solver {

        // -----------------------------------------------------------------------------
        /** 
         * @brief Exception for solver failures (e.g., numerical issues like LLT decomposition failure).
         *
         * This class represents critical solver failures that prevent optimization from proceeding.
         * It extends `std::runtime_error` to provide clear error messages when thrown.
         */
        class SolverFailure : public std::runtime_error {
        public:
            explicit SolverFailure(const std::string& message) : std::runtime_error(message) {}
        };

        // -----------------------------------------------------------------------------
        /** 
         * @brief Exception for unsuccessful optimization steps.
         *
         * This exception is thrown when an optimization step fails but does not necessarily
         * indicate a complete solver failure. It is used to signal issues that should be 
         * reported to the user while allowing the solver to attempt recovery.
         */
        class UnsuccessfulStep : public SolverFailure {
        public:
            explicit UnsuccessfulStep(const std::string& message) : SolverFailure(message) {}
        };

        // -----------------------------------------------------------------------------
        /**
         * @class SolverBase
         * @brief Abstract base class for nonlinear optimization solvers.
         *
         * This class provides an interface for solving optimization problems
         * within the SLAM framework. Derived classes implement specific solvers
         * (e.g., Gauss-Newton, Levenberg-Marquardt). 
         *
         * The solver iteratively refines a state vector to minimize a cost function
         * using nonlinear least squares techniques.
         */
        class SolverBase {
            public:
                struct Params {
                    virtual ~Params() = default;

                    /// @brief Enable verbose logging for debugging solver iterations.
                    bool verbose = false;

                    /// @brief Maximum number of solver iterations before termination.
                    unsigned int max_iterations = 100;

                    /// @brief Absolute cost threshold for convergence.
                    /// If the cost falls below this value, the solver terminates.
                    double absolute_cost_threshold = 0.0;

                    /// @brief Convergence criterion based on the absolute change in cost.
                    /// If the cost change is smaller than this threshold, the solver terminates.
                    double absolute_cost_change_threshold = 1e-4;

                    /// @brief Convergence criterion based on the relative change in cost.
                    /// If the cost reduction is below this ratio, the solver terminates.
                    double relative_cost_change_threshold = 1e-4;
                };

                // -----------------------------------------------------------------------------
                enum Termination {
                    TERMINATE_NOT_YET_TERMINATED,       ///< Solver is still running.
                    TERMINATE_STEP_UNSUCCESSFUL,       ///< Step failure (e.g., numerical instability).
                    TERMINATE_MAX_ITERATIONS,          ///< Maximum iteration count reached.
                    TERMINATE_CONVERGED_ABSOLUTE_ERROR, ///< Cost function met absolute error threshold.
                    TERMINATE_CONVERGED_ABSOLUTE_CHANGE, ///< Cost function change below absolute threshold.
                    TERMINATE_CONVERGED_RELATIVE_CHANGE, ///< Cost function change below relative threshold.
                    TERMINATE_CONVERGED_ZERO_GRADIENT,  ///< Optimization stopped due to zero gradient.
                    TERMINATE_COST_INCREASED,          ///< Cost unexpectedly increased (possible divergence).
                    TERMINATE_EXPECTED_DELTA_COST_CONVERGED, ///< Expected cost reduction is too small.
                };

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Constructs the solver with a given problem and solver parameters.
                 * 
                 * @param problem The optimization problem to solve.
                 * @param params  Solver configuration parameters.
                 */
                explicit SolverBase(slam::problem::Problem& problem, const Params& params);

                // -----------------------------------------------------------------------------
                /** @brief Default virtual destructor */
                virtual ~SolverBase() = default;

                // -----------------------------------------------------------------------------
                /** @brief Returns the reason why the solver terminated. */
                Termination terminationCause() const { return term_; }

                // -----------------------------------------------------------------------------
                /** @brief Returns the current iteration number. */
                unsigned int currentIteration() const { return curr_iteration_; }

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Runs the solver optimization process.
                 * 
                 * Calls `iterate()` iteratively until convergence criteria are met or the maximum 
                 * iteration count is reached.
                 */
                void optimize();

            protected:
                // -----------------------------------------------------------------------------
                /** 
                 * @brief Performs a single iteration of the optimization process.
                 * 
                 * - Computes the system Jacobian.
                 * - Solves for the update step.
                 * - Updates the state vector.
                 */
                void iterate();

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes and proposes an update to the state vector.
                 *
                 * @param perturbation The proposed change in the state variables.
                 * @return The computed cost after applying the perturbation.
                 */
                double proposeUpdate(const Eigen::VectorXd& perturbation);

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Accepts the proposed state update if it improves the solution.
                 */
                void acceptProposedState();

                // -----------------------------------------------------------------------------
                /**
                 * @brief Rejects the proposed state update, reverting to the previous state.
                 */
                void rejectProposedState();

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Returns a constant reference to the state vector.
                 */
                slam::problem::StateVector::ConstPtr stateVector() const { return state_vector_; }

                // -----------------------------------------------------------------------------
                /** @brief Reference to the optimization problem */
                slam::problem::Problem& problem_;

                // -----------------------------------------------------------------------------
                /** @brief Pointer to the current state vector */
                slam::problem::StateVector::Ptr state_vector_;

                // -----------------------------------------------------------------------------
                /** @brief Backup state vector for reverting to previous values */
                slam::problem::StateVector::Ptr state_vector_backup_;

                // Solver state variables
                Termination term_ = TERMINATE_NOT_YET_TERMINATED;
                unsigned int curr_iteration_ = 0;
                bool solver_converged_ = false;
                double curr_cost_ = 0.0;
                double prev_cost_ = 0.0;
                bool pending_proposed_state_ = false;

            private:
                // -----------------------------------------------------------------------------
                /** 
                 * @brief Virtual method to perform linearization, solve, and update.
                 * 
                 * This must be implemented by derived classes.
                 * @param cost The updated cost value.
                 * @param grad_norm The norm of the cost function gradient.
                 * @return True if the update was successful, false otherwise.
                 */
                virtual bool linearizeSolveAndUpdate(double& cost, double& grad_norm) = 0;

                // -----------------------------------------------------------------------------
                /** @brief Solver parameters */
                const Params params_;
        };

        // -----------------------------------------------------------------------------
        /**
         * @brief Overloads the output stream operator for `Termination` enum.
         * 
         * @param out Output stream.
         * @param T Termination reason.
         * @return The modified output stream.
         */
        std::ostream& operator<<(std::ostream& out, const SolverBase::Termination& T);

    }  // namespace solver
}  // namespace slam
