#include <iostream>
#include <iomanip>

#include "source/include/Solver/DoglegGaussNewtonSolver.hpp"
#include "source/include/Common/Timer.hpp"

namespace slam {
    namespace solver {

        // ----------------------------------------------------------------------------
        // DoglegGaussNewtonSolver
        // ----------------------------------------------------------------------------

        DoglegGaussNewtonSolver::DoglegGaussNewtonSolver(slam::problem::Problem& problem, const Params& params)
            : GaussNewtonSolver(problem, params), params_(params) {}

        // ----------------------------------------------------------------------------
        // linearizeSolveAndUpdate
        // ----------------------------------------------------------------------------

        bool DoglegGaussNewtonSolver::linearizeSolveAndUpdate(double& cost, double& grad_norm) {
            slam::common::Timer iter_timer;
            slam::common::Timer timer;
            double build_time = 0, solve_time = 0, update_time = 0;
            cost = prev_cost_;

            // Construct Gauss-Newton system
            Eigen::SparseMatrix<double> approximate_hessian;
            Eigen::VectorXd gradient_vector;

            timer.reset();
            problem_.buildGaussNewtonTerms(approximate_hessian, gradient_vector);
            grad_norm = gradient_vector.norm();
            build_time = timer.milliseconds();

            // Solve Gauss-Newton and compute Cauchy point
            timer.reset();
            Eigen::VectorXd gauss_newton_step = solveGaussNewton(approximate_hessian, gradient_vector);
            Eigen::VectorXd cauchy_step = getCauchyPoint(approximate_hessian, gradient_vector);
            solve_time = timer.milliseconds();

            // **Efficient Dogleg Step Selection**
            double gauss_newton_norm = gauss_newton_step.norm();
            double cauchy_norm = cauchy_step.norm();
            Eigen::VectorXd step;
            std::string step_type;

            if (gauss_newton_norm <= trust_region_size_) {
                step = gauss_newton_step;
                step_type = "GN";  // Gauss-Newton step
            } else if (cauchy_norm >= trust_region_size_) {
                step = (trust_region_size_ / cauchy_norm) * cauchy_step;
                step_type = "Cauchy";  // Gradient Descent step
            } else {
                double tau = (trust_region_size_ - cauchy_norm) / (gauss_newton_norm - cauchy_norm);
                step = cauchy_step + tau * (gauss_newton_step - cauchy_step);
                step_type = "Interpolated";  // Between GN and Cauchy
            }

            // Compute predicted reduction
            double predicted_reduction = predictedReduction(approximate_hessian, gradient_vector, step);

            // **Parallel Trust Region Search using TBB**
            struct SolutionCandidate {
                double cost;
                double trust_region_size;
                double actual_to_predicted_ratio;
                Eigen::VectorXd step;

                bool operator<(const SolutionCandidate& other) const {
                    return cost > other.cost;  // Lower cost is better (min-heap behavior)
                }
            };
            tbb::concurrent_priority_queue<SolutionCandidate> candidate_queue;

            timer.reset();
            tbb::parallel_for(tbb::blocked_range<size_t>(0, params_.max_shrink_steps), [&](const tbb::blocked_range<size_t>& range) {
                double local_tr_size = trust_region_size_;
                double local_actual_to_predicted_ratio = 0;  // ✅ Thread-local variable

                for (size_t i = range.begin(); i < range.end(); ++i) {
                    double proposed_cost = proposeUpdate(step);
                    double actual_reduction = prev_cost_ - proposed_cost;
                    local_actual_to_predicted_ratio = actual_reduction / predicted_reduction;  // ✅ Local variable

                    if (local_actual_to_predicted_ratio > params_.ratio_threshold_shrink) {
                        candidate_queue.push(SolutionCandidate{proposed_cost, local_tr_size, local_actual_to_predicted_ratio, step});
                        
                        if (local_actual_to_predicted_ratio > params_.ratio_threshold_grow) {
                            local_tr_size = std::min(local_tr_size * params_.grow_coeff, 1e3);
                        }
                        break;
                    }

                    local_tr_size *= params_.shrink_coeff;
                    local_tr_size = std::max(local_tr_size, 1e-7);
                }
            });
            update_time = timer.milliseconds();

            // **Select the Best Candidate Solution**
            SolutionCandidate best_candidate;
            double best_actual_to_predicted_ratio = 0;  // ✅ Store best ratio here

            if (candidate_queue.try_pop(best_candidate)) {
                acceptProposedState();
                cost = best_candidate.cost;
                trust_region_size_ = best_candidate.trust_region_size;
                best_actual_to_predicted_ratio = best_candidate.actual_to_predicted_ratio;  // ✅ Update after selection
            } else {
                rejectProposedState();
            }

            // **Parallelized Logging (Optional)**
            if (params_.verbose) {
                if (curr_iteration_ == 1) {
                    std::cout << std::setw(4) << "Iter"
                            << std::setw(12) << "Cost"
                            << std::setw(12) << "Build (ms)"
                            << std::setw(12) << "Solve (ms)"
                            << std::setw(13) << "Update (ms)"
                            << std::setw(11) << "TR Shrink"
                            << std::setw(15) << "Step Type"
                            << std::setw(15) << "Act/Pred Ratio"
                            << std::endl;
                }

                std::cout << std::setw(4) << curr_iteration_
                        << std::setw(12) << std::setprecision(5) << cost
                        << std::setw(12) << std::setprecision(3) << std::fixed << build_time
                        << std::setw(12) << std::setprecision(3) << solve_time
                        << std::setw(13) << std::setprecision(3) << update_time
                        << std::setw(11) << params_.max_shrink_steps
                        << std::setw(15) << step_type
                        << std::setw(15) << std::setprecision(3) << best_actual_to_predicted_ratio  // ✅ Use best found value
                        << std::endl;
            }

            return !candidate_queue.empty();
        }

        // ----------------------------------------------------------------------------
        // getCauchyPoint
        // ----------------------------------------------------------------------------

        Eigen::VectorXd DoglegGaussNewtonSolver::getCauchyPoint(
            const Eigen::SparseMatrix<double>& approximate_hessian,
            const Eigen::VectorXd& gradient_vector) {
            
            // Compute Hessian-vector product
            Eigen::VectorXd gHg = approximate_hessian.selfadjointView<Eigen::Upper>() * gradient_vector;
            
            // Compute quadratic form
            double g_squared = gradient_vector.squaredNorm();
            double g_transpose_H_g = gradient_vector.dot(gHg);  // More efficient than transpose() *

            // Compute alpha, preventing division by zero
            double alpha = (g_transpose_H_g > 1e-10) ? (g_squared / g_transpose_H_g) : 1.0;

            // Return the scaled negative gradient step
            return -alpha * gradient_vector;
        }

        // ----------------------------------------------------------------------------
        // predictedReduction
        // ----------------------------------------------------------------------------

        double DoglegGaussNewtonSolver::predictedReduction(
            const Eigen::SparseMatrix<double>& approximate_hessian,
            const Eigen::VectorXd& gradient_vector, const Eigen::VectorXd& step) {
            
            // Compute Hessian product
            Eigen::VectorXd H_step = approximate_hessian.selfadjointView<Eigen::Upper>() * step;

            // Compute quadratic term step^T * H * step
            double step_trans_hessian_step = step.dot(H_step);  // More efficient than transpose() *

            // Compute and return predicted reduction
            return gradient_vector.dot(step) - 0.5 * step_trans_hessian_step;
        }
    }  // namespace solver
}  // namespace slam
