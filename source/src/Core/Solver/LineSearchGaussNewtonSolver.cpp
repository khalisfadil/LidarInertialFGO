#include <iostream>
#include <iomanip>

#include "Core/Solver/LineSearchGaussNewtonSolver.hpp"
#include "Core/Common/Timer.hpp"

namespace slam {
    namespace solver {

        // ----------------------------------------------------------------------------
        // LineSearchGaussNewtonSolver
        // ----------------------------------------------------------------------------

        LineSearchGaussNewtonSolver::LineSearchGaussNewtonSolver(slam::problem::Problem& problem, 
                                                                 const Params& params)
            : GaussNewtonSolver(problem, params), params_(params) {}

        // ----------------------------------------------------------------------------
        // linearizeSolveAndUpdate
        // ----------------------------------------------------------------------------

        bool LineSearchGaussNewtonSolver::linearizeSolveAndUpdate(double& cost, double& grad_norm) {
            slam::common::Timer iter_timer, timer;
            double build_time = 0, solve_time = 0, update_time = 0;

            // Keep previous cost in case of failure
            cost = prev_cost_;

            // Construct Gauss-Newton system
            Eigen::SparseMatrix<double> approximate_hessian;
            Eigen::VectorXd gradient_vector;

            timer.reset();
            problem_.buildGaussNewtonTerms(approximate_hessian, gradient_vector);
            grad_norm = gradient_vector.norm();
            build_time = timer.milliseconds();

            // Solve system
            timer.reset();
            Eigen::VectorXd perturbation = solveGaussNewton(approximate_hessian, gradient_vector);
            solve_time = timer.milliseconds();

            // Perform backtracking line search
            timer.reset();
            double backtrack_coeff = 1.0;
            unsigned int num_backtrack = 0;
            double proposed_cost;

            while (num_backtrack < params_.max_backtrack_steps) {
                if ((proposed_cost = proposeUpdate(backtrack_coeff * perturbation)) <= prev_cost_) {
                    acceptProposedState();
                    cost = proposed_cost;
                    break;
                }
                rejectProposedState();
                backtrack_coeff *= params_.backtrack_multiplier;
                ++num_backtrack;
            }

            update_time = timer.milliseconds();

            // Logging (verbose mode)
            if (params_.verbose) {
                if (curr_iteration_ == 1) {
                    std::cout << std::fixed << std::setw(4)  << "Iter" 
                              << std::setw(12) << "Cost" 
                              << std::setw(12) << "Build (ms)"
                              << std::setw(12) << "Solve (ms)"
                              << std::setw(13) << "Update (ms)"
                              << std::setw(11) << "Total (ms)"
                              << std::setw(13) << "Step Size" << std::endl;
                }

                std::cout << std::fixed << std::setw(4)  << curr_iteration_
                          << std::setw(12) << std::setprecision(5) << cost
                          << std::setw(12) << std::setprecision(3) << build_time
                          << std::setw(12) << std::setprecision(3) << solve_time
                          << std::setw(13) << std::setprecision(3) << update_time
                          << std::setw(11) << std::setprecision(3) << iter_timer.milliseconds()
                          << std::setw(13) << std::setprecision(3) << backtrack_coeff << std::endl;
            }

            return num_backtrack < params_.max_backtrack_steps;
        }


    }  // namespace solver
}  // namespace slam
