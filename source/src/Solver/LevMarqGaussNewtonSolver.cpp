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
            double actual_to_predicted_ratio = 0;
            unsigned int num_tr_decreases = 0;

            // Keep previous cost in case of failure
            cost = curr_cost_;

            // Construct Gauss-Newton system
            Eigen::SparseMatrix<double> approximate_hessian;
            Eigen::VectorXd gradient_vector;

            timer.reset();
            problem_.buildGaussNewtonTerms(approximate_hessian, gradient_vector);
            grad_norm = gradient_vector.norm();
            build_time = timer.milliseconds();

            // Perform LM Search
            for (unsigned int num_backtrack = 0; num_backtrack < params_.max_shrink_steps; ++num_backtrack) {
                // Solve system
                timer.reset();
                Eigen::VectorXd lev_marq_step;
                try {
                    lev_marq_step = solveLevMarq(approximate_hessian, gradient_vector, diag_coeff_);
                } catch (const slam::solver::DecompFailure&) {
                    solve_time += timer.milliseconds();
                    return false;  // Decomposition failed, exit optimization step
                }
                solve_time += timer.milliseconds();

                // Test new cost
                timer.reset();
                double proposed_cost = proposeUpdate(lev_marq_step);
                double actual_reduction = curr_cost_ - proposed_cost;
                double predicted_reduction = predictedReduction(approximate_hessian, gradient_vector, lev_marq_step);
                actual_to_predicted_ratio = actual_reduction / predicted_reduction;

                if (actual_to_predicted_ratio > params_.ratio_threshold) {
                    acceptProposedState();
                    cost = proposed_cost;
                    diag_coeff_ = std::max(diag_coeff_ * params_.shrink_coeff, 1e-7);
                    update_time += timer.milliseconds();
                    break;
                } else {
                    rejectProposedState();
                    diag_coeff_ = std::min(diag_coeff_ * params_.grow_coeff, 1e7);
                    num_tr_decreases++;
                    update_time += timer.milliseconds();
                }
            }

            return true;
        }

        // ----------------------------------------------------------------------------
        // solveLevMarq
        // ----------------------------------------------------------------------------

        Eigen::VectorXd LevMarqGaussNewtonSolver::solveLevMarq(
            const Eigen::SparseMatrix<double>& approximate_hessian,
            const Eigen::VectorXd& gradient_vector, double diagonal_coeff) {
            
            Eigen::SparseMatrix<double> mod_hessian = approximate_hessian;
            Eigen::VectorXd lev_marq_step;

            // Modify diagonal without altering original matrix
            Eigen::DiagonalMatrix<double, Eigen::Dynamic> diag_adjustment(approximate_hessian.rows());
            diag_adjustment.diagonal() = approximate_hessian.diagonal() * diagonal_coeff;

            mod_hessian += diag_adjustment;

            try {
                lev_marq_step = solveGaussNewton(mod_hessian, gradient_vector);
            } catch (const DecompFailure&) {
                throw;  // Propagate failure upwards
            }

            return lev_marq_step;
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
