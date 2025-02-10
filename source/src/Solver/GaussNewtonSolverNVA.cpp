#include <iostream>
#include <iomanip>
#include <Eigen/Cholesky>

#include "source/include/Solver/GaussNewtonSolverNVA.hpp"
#include "source/include/Common/Timer.hpp"

namespace slam {
    namespace solver {

        // ----------------------------------------------------------------------------
        // GaussNewtonSolverNVA
        // ----------------------------------------------------------------------------

        GaussNewtonSolverNVA::GaussNewtonSolverNVA(slam::problem::Problem& problem, const Params& params)
            : GaussNewtonSolver(problem, params), params_(params) {
            // Initialize solver only when required
            if (!hessian_solver_) {
                hessian_solver_ = std::make_shared<SolverType>();
            }
        }

        // ----------------------------------------------------------------------------
        // linearizeSolveAndUpdate
        // ----------------------------------------------------------------------------

        bool GaussNewtonSolverNVA::linearizeSolveAndUpdate(double& cost, double& grad_norm) {
            slam::common::Timer iter_timer;
            slam::common::Timer timer;
            double build_time = 0, solve_time = 0, update_time = 0;

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

            // Apply line search if enabled
            if (params_.line_search) {
                const double expected_delta_cost = 0.5 * gradient_vector.transpose() * perturbation;
                if (expected_delta_cost < 0.0) {
                    throw slam::solver::SolverFailure("[GaussNewtonSolverNVA::linearizeSolveAndUpdate] Expected delta cost must be >= 0.0.");
                }
                if (expected_delta_cost < 1.0e-5 || fabs(expected_delta_cost / cost) < 1.0e-7) {
                    solver_converged_ = true;
                    term_ = TERMINATE_EXPECTED_DELTA_COST_CONVERGED;
                } else {
                    double alpha = 1.0;
                    for (unsigned int j = 0; j < 3; ++j) {
                        timer.reset();
                        cost = proposeUpdate(alpha * perturbation);
                        update_time += timer.milliseconds();

                        if (cost <= prev_cost_) {
                            acceptProposedState();
                            break;
                        } else {
                            cost = prev_cost_;
                            rejectProposedState();
                        }
                        alpha *= 0.5;
                    }
                }
            } else {
                // Standard Gauss-Newton update
                timer.reset();
                cost = proposeUpdate(perturbation);
                acceptProposedState();
                update_time = timer.milliseconds();
            }

            // Logging (verbose mode)
            if (params_.verbose) {
                if (curr_iteration_ == 1) {
                    std::cout << std::setw(4) << "Iter"
                              << std::setw(12) << "Cost"
                              << std::setw(12) << "Build (ms)"
                              << std::setw(12) << "Solve (ms)"
                              << std::setw(13) << "Update (ms)"
                              << std::setw(11) << "Total (ms)"
                              << std::endl;
                }

                std::cout << std::setw(4) << curr_iteration_
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

        Eigen::VectorXd GaussNewtonSolverNVA::solveGaussNewton(
            const Eigen::SparseMatrix<double>& approximate_hessian,
            const Eigen::VectorXd& gradient_vector) {
            
            if (!pattern_initialized_) {
                hessian_solver_->analyzePattern(approximate_hessian);
                pattern_initialized_ = params_.reuse_previous_pattern;
            }

            hessian_solver_->factorize(approximate_hessian);

            if (hessian_solver_->info() != Eigen::Success) {
                throw slam::solver::SolverFailure(
                    "[GaussNewtonSolverNVA::solveGaussNewton] Eigen LLT decomposition failed. The Hessian matrix may be ill-conditioned.");
            }

            return hessian_solver_->solve(gradient_vector);
        }

    }  // namespace solver
}  // namespace slam
