#pragma once

#include <Eigen/Core>
#include <memory>

#include "Core/Solver/GaussNewtonSolver.hpp"

namespace slam {
    namespace solver {

        // -----------------------------------------------------------------------------
        /**
         * @class GaussNewtonSolverNVA
         * @brief Gauss-Newton solver with optional line search.
         */
        class GaussNewtonSolverNVA : public GaussNewtonSolver {

            public:
            
                struct Params : public GaussNewtonSolver::Params {
                    /// Whether to use line search (default: false)
                    bool line_search = false;
                };

                // -----------------------------------------------------------------------------
                explicit GaussNewtonSolverNVA(slam::problem::Problem& problem, const Params& params);

            protected:
                Eigen::VectorXd solveGaussNewton(
                    const Eigen::SparseMatrix<double>& approximate_hessian,
                    const Eigen::VectorXd& gradient_vector);

            private:
                using SolverType = Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper>;

                std::shared_ptr<SolverType> solver() { return hessian_solver_; }
                bool linearizeSolveAndUpdate(double& cost, double& grad_norm) override;

                std::shared_ptr<SolverType> hessian_solver_ = nullptr;
                bool pattern_initialized_ = false;

                const Params params_;
        };

    }  // namespace solver
}  // namespace slam
