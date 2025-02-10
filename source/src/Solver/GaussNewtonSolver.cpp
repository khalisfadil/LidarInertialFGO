#include <iomanip>
#include <iostream>
#include <Eigen/Cholesky>

#include "source/include/Solver/GaussNewtonSolver.hpp"
#include "source/include/Common/Timer.hpp"

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
            slam::common::Timer iter_timer;
            slam::common::Timer timer;
            double build_time = 0, solve_time = 0, update_time = 0;

            Eigen::SparseMatrix<double> approximate_hessian;
            Eigen::VectorXd gradient_vector;

            // Construct system of equations
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
                    std::cout << std::right << std::setw(4)  << "iter"
                              << std::right << std::setw(12) << "cost"
                              << std::right << std::setw(12) << "build (ms)"
                              << std::right << std::setw(12) << "solve (ms)"
                              << std::right << std::setw(13) << "update (ms)"
                              << std::right << std::setw(11) << "time (ms)"
                              << std::endl;
                }

                std::cout << std::right << std::setw(4)  << curr_iteration_
                          << std::right << std::setw(12) << std::setprecision(5) << cost
                          << std::right << std::setw(12) << std::setprecision(3) << std::fixed << build_time << std::resetiosflags(std::ios::fixed)
                          << std::right << std::setw(12) << std::setprecision(3) << std::fixed << solve_time << std::resetiosflags(std::ios::fixed)
                          << std::right << std::setw(13) << std::setprecision(3) << std::fixed << update_time << std::resetiosflags(std::ios::fixed)
                          << std::right << std::setw(11) << std::setprecision(3) << std::fixed << iter_timer.milliseconds() << std::resetiosflags(std::ios::fixed)
                          << std::endl;
            }

            return true;
        }

        // ----------------------------------------------------------------------------
        // solveGaussNewton
        // ----------------------------------------------------------------------------

        Eigen::VectorXd GaussNewtonSolver::solveGaussNewton(
            const Eigen::SparseMatrix<double>& approximate_hessian,
            const Eigen::VectorXd& gradient_vector
        ) {
            // Initialize pattern if it's the first iteration
            if (!pattern_initialized_) {
                hessian_solver_->analyzePattern(approximate_hessian);
                if (params_.reuse_previous_pattern) pattern_initialized_ = true;
            }

            // Factorize the Hessian using Cholesky decomposition
            hessian_solver_->factorize(approximate_hessian);

            // Check if the factorization succeeded
            if (hessian_solver_->info() != Eigen::Success) {
                throw DecompFailure(
                    "[GaussNewtonSolver::solveGaussNewton] Eigen LLT decomposition failed. "
                    "Possible causes: ill-conditioned Hessian or non-positive-definite matrix."
                );
            }

            // Solve the linear system H * x = -gradient
            return hessian_solver_->solve(gradient_vector);
        }

    }  // namespace solver
}  // namespace slam
