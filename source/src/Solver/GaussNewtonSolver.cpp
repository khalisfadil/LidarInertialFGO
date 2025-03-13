#include <iomanip>
#include <iostream>
#include <Eigen/Cholesky>

#include "Solver/GaussNewtonSolver.hpp"
#include "Common/Timer.hpp"

namespace slam {
    namespace solver {

        // ----------------------------------------------------------------------------
        // GaussNewtonSolver Constructor
        // ----------------------------------------------------------------------------
        
        GaussNewtonSolver::GaussNewtonSolver(slam::problem::Problem& problem, const Params& params)
            : SolverBase(problem, params), params_(params) {}

        // ----------------------------------------------------------------------------
        // linearizeSolveAndUpdate
        // ----------------------------------------------------------------------------

        bool GaussNewtonSolver::linearizeSolveAndUpdate(double& cost, double& grad_norm) {
            slam::common::Timer iter_timer, timer;
            double build_time = 0, solve_time = 0, update_time = 0;

            // Initialize Hessian and gradient vector
            Eigen::SparseMatrix<double> approximate_hessian;
            Eigen::VectorXd gradient_vector;

            // Construct Gauss-Newton system
            timer.reset();
            problem_.buildGaussNewtonTerms(approximate_hessian, gradient_vector);
            grad_norm = gradient_vector.norm();
            build_time = timer.milliseconds();

            // Solve system
            timer.reset();
            Eigen::VectorXd perturbation = solveGaussNewton(approximate_hessian, gradient_vector);
            solve_time = timer.milliseconds();

            // Apply update
            timer.reset();
            cost = proposeUpdate(perturbation);
            acceptProposedState();
            update_time = timer.milliseconds();

            // Print debug info if verbose mode is enabled
            if (params_.verbose) {
                if (curr_iteration_ == 1) {
                    std::cout << std::setw(4)  << "Iter"
                            << std::setw(12) << "Cost"
                            << std::setw(12) << "Build (ms)"
                            << std::setw(12) << "Solve (ms)"
                            << std::setw(13) << "Update (ms)"
                            << std::setw(11) << "Time (ms)"
                            << std::endl;
                }

                std::cout << std::setw(4)  << curr_iteration_
                        << std::setw(12) << std::setprecision(5) << cost
                        << std::setw(12) << std::setprecision(3) << std::fixed << build_time
                        << std::setw(12) << std::setprecision(3) << solve_time
                        << std::setw(13) << std::setprecision(3) << update_time
                        << std::setw(11) << std::setprecision(3) << iter_timer.milliseconds()
                        << std::endl;
            }

            return true;
        }

        // ----------------------------------------------------------------------------
        // solveGaussNewton
        // ----------------------------------------------------------------------------

        Eigen::VectorXd GaussNewtonSolver::solveGaussNewton(
            const Eigen::SparseMatrix<double>& approximate_hessian,
            const Eigen::VectorXd& gradient_vector) {
            
            // Initialize sparsity pattern analysis on the first iteration
            if (!pattern_initialized_) {
                hessian_solver_->analyzePattern(approximate_hessian);
                if (params_.reuse_previous_pattern) pattern_initialized_ = true;
            }

            // Perform Cholesky factorization of the approximate Hessian
            hessian_solver_->factorize(approximate_hessian);

            // Check if the factorization succeeded
            if (hessian_solver_->info() != Eigen::Success) {
                throw DecompFailure(
                    "[GaussNewtonSolver::solveGaussNewton] Eigen LLT decomposition failed. "
                    "Likely causes: ill-conditioned Hessian or non-positive-definite matrix. "
                    "Consider adding a prior or verifying the problem formulation."
                );
            }

            // TODO: Consider checking the condition number for additional numerical stability analysis.

            // Solve the linear system H * x = -gradient using the Cholesky factorization
            return hessian_solver_->solve(gradient_vector);
        }

    }  // namespace solver
}  // namespace slam
