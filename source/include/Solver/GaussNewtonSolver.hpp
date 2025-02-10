#pragma once

#include <Eigen/Core>
#include <memory>

#include "source/include/Problem/Problem.hpp"
#include "source/include/Solver/SolverBase.hpp"

namespace slam {
    namespace solver {

        // -----------------------------------------------------------------------------
        /**
         * @class DecompFailure
         * @brief Exception class for decomposition failures.
         *
         * This exception is thrown when Cholesky factorization fails, which usually
         * happens due to numerical instability, ill-conditioning, or the system
         * not being positive semi-definite.
         */
        class DecompFailure : public SolverFailure {
        public:
            explicit DecompFailure(const std::string& msg) : SolverFailure(msg) {}
        };

        // -----------------------------------------------------------------------------
        /**
         * @class GaussNewtonSolver
         * @brief Implements the Gauss-Newton optimization algorithm.
         *
         * - Solves non-linear least squares problems via **Cholesky factorization**.
         * - Supports **sparsity pattern reuse** for faster convergence.
         * - Handles **Hessian ill-conditioning** with failure detection.
         */
        class GaussNewtonSolver : public SolverBase {
        public:
            using Ptr = std::shared_ptr<GaussNewtonSolver>;
            using ConstPtr = std::shared_ptr<const GaussNewtonSolver>;

            // -----------------------------------------------------------------------------
            /**
             * @struct Params
             * @brief Configuration parameters for the Gauss-Newton solver.
             */
            struct Params : public SolverBase::Params {
                /** \brief Enables reuse of the sparsity pattern in the Hessian. */
                bool reuse_previous_pattern = true;
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs the Gauss-Newton solver.
             * @param problem Reference to the optimization problem.
             * @param params Solver parameters.
             */
            GaussNewtonSolver(slam::problem::Problem& problem, const Params& params);

            /** @brief Returns shared pointer to the Hessian solver */
            std::shared_ptr<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper>> getHessianSolver() const {
                return hessian_solver_;
            }

        protected:
            // -----------------------------------------------------------------------------
            /**
             * @brief Solves the Gauss-Newton system: `Hessian * x = gradient`
             * @param approximate_hessian The Hessian matrix (upper triangular part).
             * @param gradient_vector The gradient vector.
             * @return The computed perturbation vector.
             */
            Eigen::VectorXd solveGaussNewton(
                const Eigen::SparseMatrix<double>& approximate_hessian,
                const Eigen::VectorXd& gradient_vector
            );

        private:
            using SolverType = Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper>;

            // -----------------------------------------------------------------------------
            /** \brief Implements one iteration of the Gauss-Newton algorithm */
            bool linearizeSolveAndUpdate(double& cost, double& grad_norm) override;

            // -----------------------------------------------------------------------------
            /** \brief Shared pointer for the Hessian factorization solver */
            std::shared_ptr<SolverType> hessian_solver_ = std::make_shared<SolverType>();

            // -----------------------------------------------------------------------------
            /** \brief Flag indicating whether the Hessian sparsity pattern is initialized */
            bool pattern_initialized_ = false;

            // -----------------------------------------------------------------------------
            /** \brief Solver configuration parameters */
            const Params params_;

            // -----------------------------------------------------------------------------
            /** @brief Allows `Covariance` class to access internal solver object */
            friend class Covariance;
        };

    }  // namespace solver
}  // namespace slam
