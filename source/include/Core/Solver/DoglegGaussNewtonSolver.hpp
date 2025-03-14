#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <memory>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_priority_queue.h>

#include "Core/Solver/GaussNewtonSolver.hpp"

namespace slam {
    namespace solver {

        // -----------------------------------------------------------------------------
        /**
         * @class DoglegGaussNewtonSolver
         * @brief Gauss-Newton solver using the Powell Dogleg method.
         *
         * - Combines **Gauss-Newton** and **gradient descent** for robust trust region handling.
         * - Adapts trust region size dynamically for stability.
         */
        class DoglegGaussNewtonSolver : public GaussNewtonSolver {
        public:
            struct Params : public GaussNewtonSolver::Params {
                /// Minimum ratio of actual to predicted cost reduction, shrink trust region if lower (range: 0.0-1.0)
                double ratio_threshold_shrink = 0.25;
                /// Grow trust region if ratio of actual to predicted cost reduction above this (range: 0.0-1.0)
                double ratio_threshold_grow = 0.75;
                /// Amount to shrink by (range: <1.0)
                double shrink_coeff = 0.5;
                /// Amount to grow by (range: >1.0)
                double grow_coeff = 3.0;
                /// Maximum number of times to shrink trust region before giving up
                unsigned int max_shrink_steps = 50;
            };

            explicit DoglegGaussNewtonSolver(slam::problem::Problem& problem, const Params& params);

        private:

            // -----------------------------------------------------------------------------
            /**
             * @brief Performs the linearization, solves the Gauss-Newton system, and updates the state.
             * 
             * This function constructs the Gauss-Newton system, computes the gradient norm, and 
             * determines the optimal step direction using the Dogleg trust region method. It evaluates 
             * the proposed step, updates the trust region size accordingly, and determines if the 
             * optimization step was successful.
             * 
             * @param[out] cost Updated cost after applying the optimization step.
             * @param[out] grad_norm Computed gradient norm for convergence checks.
             * @return True if the optimization step is successful, false otherwise.
             */
            bool linearizeSolveAndUpdate(double& cost, double& grad_norm) override;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the Cauchy point (gradient descent direction).
             * @return Cauchy point step direction.
             */
            Eigen::VectorXd getCauchyPoint(
                const Eigen::SparseMatrix<double>& approximate_hessian,
                const Eigen::VectorXd& gradient_vector);

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the predicted cost reduction.
             */
            double predictedReduction(const Eigen::SparseMatrix<double>& approximate_hessian,
                const Eigen::VectorXd& gradient_vector, const Eigen::VectorXd& step);

            double trust_region_size_ = 1.0;
            const Params params_;
        };

    }  // namespace solver
}  // namespace slam
