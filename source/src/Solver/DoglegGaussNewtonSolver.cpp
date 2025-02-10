
#include <iostream>
#include <iomanip>
#include <Eigen/Cholesky>

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
            double actual_to_predicted_ratio = 0;
            unsigned int num_shrink_steps = 0;

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
            Eigen::VectorXd gauss_newton_step = solveGaussNewton(approximate_hessian, gradient_vector);
            Eigen::VectorXd cauchy_step = getCauchyPoint(approximate_hessian, gradient_vector);
            solve_time = timer.milliseconds();

            Eigen::VectorXd step;
            std::string step_type;

            // Choose step based on trust region
            double gauss_newton_norm = gauss_newton_step.norm();
            double cauchy_norm = cauchy_step.norm();

            if (gauss_newton_norm <= trust_region_size_) {
                step = gauss_newton_step;
                step_type = "GN";  // Gauss-Newton
            } else if (cauchy_norm >= trust_region_size_) {
                step = (trust_region_size_ / cauchy_norm) * cauchy_step;
                step_type = "Cauchy";  // Gradient Descent
            } else {
                double tau = (trust_region_size_ - cauchy_norm) / (gauss_newton_norm - cauchy_norm);
                step = cauchy_step + tau * (gauss_newton_step - cauchy_step);
                step_type = "Interpolated";  // Between GN and Cauchy
            }

            // Compute predicted reduction
            double predicted_reduction = predictedReduction(approximate_hessian, gradient_vector, step);

            // Perform trust region update
            timer.reset();
            for (; num_shrink_steps < params_.max_shrink_steps; ++num_shrink_steps) {
                double proposed_cost = proposeUpdate(step);
                double actual_reduction = prev_cost_ - proposed_cost;
                actual_to_predicted_ratio = actual_reduction / predicted_reduction;

                if (actual_to_predicted_ratio > params_.ratio_threshold_shrink) {
                    acceptProposedState();
                    cost = proposed_cost;

                    if (actual_to_predicted_ratio > params_.ratio_threshold_grow) {
                        trust_region_size_ = std::min(trust_region_size_ * params_.grow_coeff, 1e3);
                    }
                    break;
                }

                trust_region_size_ *= params_.shrink_coeff;
                trust_region_size_ = std::max(trust_region_size_, 1e-7);
                rejectProposedState();
            }

            update_time = timer.milliseconds();

            // Logging (verbose mode)
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
                          << std::setw(11) << num_shrink_steps
                          << std::setw(15) << step_type
                          << std::setw(15) << std::setprecision(3) << actual_to_predicted_ratio
                          << std::endl;
            }

            return num_shrink_steps < params_.max_shrink_steps;
        }

        // ----------------------------------------------------------------------------
        // getCauchyPoint
        // ----------------------------------------------------------------------------

        Eigen::VectorXd DoglegGaussNewtonSolver::getCauchyPoint(
            const Eigen::SparseMatrix<double>& approximate_hessian,
            const Eigen::VectorXd& gradient_vector) {
            
            Eigen::VectorXd gHg = approximate_hessian.selfadjointView<Eigen::Upper>() * gradient_vector;
            double g_squared = gradient_vector.squaredNorm();
            double g_transpose_H_g = gradient_vector.transpose() * gHg;

            // Avoid division by zero
            double alpha = (g_transpose_H_g > 1e-10) ? (g_squared / g_transpose_H_g) : 1.0;
            
            return -alpha * gradient_vector;
        }

        // ----------------------------------------------------------------------------
        // predictedReduction
        // ----------------------------------------------------------------------------

        double DoglegGaussNewtonSolver::predictedReduction(
            const Eigen::SparseMatrix<double>& approximate_hessian,
            const Eigen::VectorXd& gradient_vector, const Eigen::VectorXd& step) {
            
            double step_trans_hessian_step =
                (step.transpose() * (approximate_hessian.selfadjointView<Eigen::Upper>() * step)).value();

            return gradient_vector.transpose() * step - 0.5 * step_trans_hessian_step;
        }
    }  // namespace solver
}  // namespace slam
