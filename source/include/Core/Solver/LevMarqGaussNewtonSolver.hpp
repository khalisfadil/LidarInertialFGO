#pragma once

#include <Eigen/Core>
#include <memory>

#include "Core/Solver/GaussNewtonSolver.hpp"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_priority_queue.h>

namespace slam {
    namespace solver {

        // ----------------------------------------------------------------------------
        /**
         * @class LevMarqGaussNewtonSolver
         * @brief Implements a Levenberg-Marquardt solver built on the Gauss-Newton method.
         * 
         * This solver introduces a trust-region approach using the Levenberg-Marquardt algorithm.
         * It dynamically adjusts the trust-region size based on the ratio of actual-to-predicted 
         * reduction in cost. This method ensures stability in non-convex optimization problems.
         */
        class LevMarqGaussNewtonSolver : public GaussNewtonSolver {
            public:
                // ----------------------------------------------------------------------------
                /**
                 * @struct Params
                 * @brief Configuration parameters for the Levenberg-Marquardt solver.
                 * 
                 * This structure extends `GaussNewtonSolver::Params` and provides additional 
                 * parameters for trust-region adaptation in the LM optimization process.
                 */
                struct Params : public GaussNewtonSolver::Params {
                    double ratio_threshold = 0.25;  ///< @brief Minimum ratio of actual-to-predicted reduction (range: 0.0 - 1.0).
                    double shrink_coeff = 0.1;      ///< @brief Factor to shrink the trust region when the step fails (value < 1.0).
                    double grow_coeff = 10.0;      ///< @brief Factor to expand the trust region when the step succeeds (value > 1.0).
                    unsigned int max_shrink_steps = 50; ///< @brief Maximum allowed iterations for trust region reduction.
                };

                // ----------------------------------------------------------------------------
                /**
                 * @brief Constructs the Levenberg-Marquardt Gauss-Newton solver.
                 * 
                 * @param problem Reference to the optimization problem.
                 * @param params Configuration parameters for the solver.
                 */
                explicit LevMarqGaussNewtonSolver(slam::problem::Problem& problem, const Params& params);

            private:
                // ----------------------------------------------------------------------------
                /**
                 * @brief Performs a single optimization step: linearization, solving, and state update.
                 * 
                 * This method constructs the Gauss-Newton system, applies the Levenberg-Marquardt
                 * trust-region strategy, solves for the optimal step, and updates the state.
                 * If the step is rejected, it shrinks the trust region and retries until success or
                 * until reaching the maximum number of shrink steps.
                 * 
                 * @param[out] cost Updated cost after applying the optimization step.
                 * @param[out] grad_norm Norm of the gradient vector after solving.
                 * @return `true` if the step is successfully applied, `false` if optimization fails.
                 */
                bool linearizeSolveAndUpdate(double& cost, double& grad_norm) override;

                // ----------------------------------------------------------------------------
                /**
                 * @brief Solves the Levenberg-Marquardt linear system.
                 * 
                 * Computes the solution to the LM-adjusted Gauss-Newton system:
                 * \f[
                 * (J^T J + \lambda D) x = -J^T r
                 * \f]
                 * where \(D\) is a diagonal scaling matrix derived from the approximate Hessian.
                 * 
                 * @param approximate_hessian The approximate Hessian matrix (J^T * J).
                 * @param gradient_vector The gradient vector (J^T * r).
                 * @param diagonal_coeff The diagonal scaling factor (λ).
                 * @return The computed step direction for the optimization update.
                 */
                Eigen::VectorXd solveLevMarq(const Eigen::SparseMatrix<double>& approximate_hessian,
                                             const Eigen::VectorXd& gradient_vector, double diagonal_coeff);

                // ----------------------------------------------------------------------------
                /**
                 * @brief Computes the predicted reduction in cost based on the quadratic model.
                 * 
                 * The Levenberg-Marquardt method predicts the potential cost reduction using:
                 * \f[
                 * \rho = \frac{\text{actual reduction}}{\text{predicted reduction}}
                 * \f]
                 * where:
                 * \f[
                 * \text{predicted reduction} = g^T s - \frac{1}{2} s^T H s
                 * \f]
                 * 
                 * @param approximate_hessian The approximate Hessian matrix (J^T * J).
                 * @param gradient_vector The gradient vector (J^T * r).
                 * @param step The computed optimization step.
                 * @return The predicted reduction value.
                 */
                double predictedReduction(const Eigen::SparseMatrix<double>& approximate_hessian,
                                          const Eigen::VectorXd& gradient_vector, const Eigen::VectorXd& step);

                // ----------------------------------------------------------------------------
                /**
                 * @brief Trust-region damping factor (λ) for Levenberg-Marquardt updates.
                 * 
                 * This factor controls the balance between Gauss-Newton and gradient descent.
                 * A small value approaches Gauss-Newton, while a larger value moves towards
                 * gradient descent for robustness.
                 */
                double diag_coeff_ = 1e-7;

                // ----------------------------------------------------------------------------
                /**
                 * @brief Reference to solver parameters to avoid unnecessary copies.
                 */
                const Params& params_;
        };

    }  // namespace solver
}  // namespace slam
