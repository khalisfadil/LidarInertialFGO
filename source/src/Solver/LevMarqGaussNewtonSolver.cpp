#include <iostream>
#include <iomanip>

#include "source/include/Solver/LevMarqGaussNewtonSolver.hpp"
#include "source/include/Common/Timer.hpp"

namespace slam {
    namespace solver {

        // ----------------------------------------------------------------------------
        // LevMarqGaussNewtonSolver
        // ----------------------------------------------------------------------------

        LevMarqGaussNewtonSolver::LevMarqGaussNewtonSolver(slam::problem::Problem& problem, 
                                                           const Params& params)
            : GaussNewtonSolver(problem, params), params_(params) {}

        // ----------------------------------------------------------------------------
        // linearizeSolveAndUpdate
        // ----------------------------------------------------------------------------

        bool LevMarqGaussNewtonSolver::linearizeSolveAndUpdate(double& cost, double& grad_norm) {
            slam::common::Timer iter_timer;
            slam::common::Timer timer;

            double build_time = 0, solve_time = 0, update_time = 0;
            cost = curr_cost_;

            // Construct Gauss-Newton system
            Eigen::SparseMatrix<double> approximate_hessian;
            Eigen::VectorXd gradient_vector;

            timer.reset();
            problem_.buildGaussNewtonTerms(approximate_hessian, gradient_vector);
            grad_norm = gradient_vector.norm();
            build_time = timer.milliseconds();

            // Priority queue for selecting the best solution
            struct SolutionCandidate {
                double cost;
                double diag_coeff;
                Eigen::VectorXd step;
                
                bool operator<(const SolutionCandidate& other) const {
                    return cost > other.cost;  // Lower cost is better (min-heap behavior)
                }
            };
            tbb::concurrent_priority_queue<SolutionCandidate> candidate_queue;

            // Perform LM Search using TBB with local copies of diag_coeff_
            tbb::parallel_for(tbb::blocked_range<unsigned int>(0, params_.max_shrink_steps), [&](const tbb::blocked_range<unsigned int>& range) {
                double local_diag_coeff = diag_coeff_;
                for (unsigned int num_backtrack = range.begin(); num_backtrack < range.end(); ++num_backtrack) {
                    
                    slam::common::Timer local_timer;
                    Eigen::VectorXd lev_marq_step;

                    try {
                        lev_marq_step = solveLevMarq(approximate_hessian, gradient_vector, local_diag_coeff);
                    } catch (const slam::solver::DecompFailure&) {
                        solve_time += local_timer.milliseconds();
                        return;
                    }
                    solve_time += local_timer.milliseconds();

                    // Test new cost
                    local_timer.reset();
                    double proposed_cost = proposeUpdate(lev_marq_step);
                    double actual_reduction = curr_cost_ - proposed_cost;
                    double predicted_reduction = predictedReduction(approximate_hessian, gradient_vector, lev_marq_step);
                    double local_actual_to_predicted_ratio = actual_reduction / predicted_reduction;

                    if (local_actual_to_predicted_ratio > params_.ratio_threshold) {
                        local_diag_coeff = std::max(local_diag_coeff * params_.shrink_coeff, 1e-7);
                    } else {
                        local_diag_coeff = std::min(local_diag_coeff * params_.grow_coeff, 1e7);
                    }

                    // Push candidate solution to the priority queue
                    candidate_queue.push(SolutionCandidate{proposed_cost, local_diag_coeff, lev_marq_step});
                    update_time += local_timer.milliseconds();
                }
            });

            // Select the best solution
            SolutionCandidate best_candidate;
            if (candidate_queue.try_pop(best_candidate)) {
                acceptProposedState();
                cost = best_candidate.cost;
                diag_coeff_ = best_candidate.diag_coeff;
            } else {
                rejectProposedState();
            }

            // Logging
            if (params_.verbose) {
                if (curr_iteration_ == 1) {
                    std::cout << std::right << std::setw( 4) << "Iter"
                            << std::right << std::setw(12) << "Cost"
                            << std::right << std::setw(12) << "Build (ms)"
                            << std::right << std::setw(12) << "Solve (ms)"
                            << std::right << std::setw(13) << "Update (ms)"
                            << std::right << std::setw(11) << "Total (ms)"
                            << std::endl;
                }

                std::cout << std::right << std::setw( 4) << curr_iteration_
                        << std::right << std::setw(12) << std::setprecision(5) << cost
                        << std::right << std::setw(12) << std::fixed << std::setprecision(3) << build_time << std::resetiosflags(std::ios::fixed)
                        << std::right << std::setw(12) << std::fixed << std::setprecision(3) << solve_time << std::resetiosflags(std::ios::fixed)
                        << std::right << std::setw(13) << std::fixed << std::setprecision(3) << update_time << std::resetiosflags(std::ios::fixed)
                        << std::right << std::setw(11) << std::fixed << std::setprecision(3) << iter_timer.milliseconds() << std::resetiosflags(std::ios::fixed)
                        << std::endl;
            }

            return !candidate_queue.empty();
        }


        // ----------------------------------------------------------------------------
        // solveLevMarq
        // ----------------------------------------------------------------------------

        Eigen::VectorXd LevMarqGaussNewtonSolver::solveLevMarq(
            const Eigen::SparseMatrix<double>& approximate_hessian,
            const Eigen::VectorXd& gradient_vector, double diagonal_coeff) {
            
            Eigen::SparseMatrix<double> mod_hessian = approximate_hessian;
            const int rows = mod_hessian.rows();

            // Parallel diagonal modification using TBB
            tbb::parallel_for(tbb::blocked_range<int>(0, rows), [&](const tbb::blocked_range<int>& range) {
                for (int k = range.begin(); k < range.end(); ++k) {
                    for (Eigen::SparseMatrix<double>::InnerIterator it(mod_hessian, k); it; ++it) {
                        if (it.row() == it.col()) {
                            it.valueRef() += diagonal_coeff * it.value();
                        }
                    }
                }
            });

            try {
                return solveGaussNewton(mod_hessian, gradient_vector);
            } catch (const DecompFailure&) {
                throw;  // Directly propagate exception
            }
        }

        // ----------------------------------------------------------------------------
        // predictedReduction
        // ----------------------------------------------------------------------------

        double LevMarqGaussNewtonSolver::predictedReduction(
            const Eigen::SparseMatrix<double>& approximate_hessian,
            const Eigen::VectorXd& gradient_vector, const Eigen::VectorXd& step) {
            
            double step_trans_hessian_step =
                (step.transpose() * (approximate_hessian.selfadjointView<Eigen::Upper>() * step)).value();

            return gradient_vector.transpose() * step - 0.5 * step_trans_hessian_step;
        }

    }  // namespace solver
}  // namespace slam
