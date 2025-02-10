#pragma once

#include "source/include/Solver/GaussNewtonSolver.hpp"

namespace slam {
    namespace solver {

        // ----------------------------------------------------------------------------
        /**
         * @class LevMarqGaussNewtonSolver
         * @brief Levenberg-Marquardt Gauss-Newton solver with trust region adaptation.
         */
        class LevMarqGaussNewtonSolver : public GaussNewtonSolver {
            public:
                // ----------------------------------------------------------------------------
                struct Params : public GaussNewtonSolver::Params {
                    /// Trust-region update parameters
                    double ratio_threshold = 0.25;  ///< Min ratio of actual-to-predicted reduction (0.0 - 1.0)
                    double shrink_coeff = 0.1;      ///< Shrink factor for trust region (<1.0)
                    double grow_coeff = 10.0;      ///< Growth factor for trust region (>1.0)
                    unsigned int max_shrink_steps = 50; ///< Max iterations for reducing trust region
                };

                // ----------------------------------------------------------------------------
                explicit LevMarqGaussNewtonSolver(slam::problem::Problem& problem, const Params& params);

            private:
                // ----------------------------------------------------------------------------
                bool linearizeSolveAndUpdate(double& cost, double& grad_norm) override;

                // ----------------------------------------------------------------------------
                /**
                 * @brief Solve the Levenberg–Marquardt system: (J^T * J + λ * diag(J^T * J)) * x = -J^T * r
                 */
                Eigen::VectorXd solveLevMarq(const Eigen::SparseMatrix<double>& approximate_hessian,
                                             const Eigen::VectorXd& gradient_vector, double diagonal_coeff);

                // ----------------------------------------------------------------------------
                /**
                 * @brief Compute the predicted reduction of cost.
                 */
                double predictedReduction(const Eigen::SparseMatrix<double>& approximate_hessian,
                                          const Eigen::VectorXd& gradient_vector, const Eigen::VectorXd& step);

                // ----------------------------------------------------------------------------
                /// Trust-region scaling factor (λ in LM)
                double diag_coeff_ = 1e-7;

                // ----------------------------------------------------------------------------
                /// Reference to solver parameters (avoids unnecessary copies)
                const Params& params_;
        };

    }  // namespace solver
}  // namespace slam
