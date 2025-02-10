#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "source/include/Problem/Problem.hpp"
#include "source/include/Problem/StateVector.hpp"
#include "source/include/Solver/GaussNewtonSolver.hpp"

namespace slam {
    namespace solver {

        // -----------------------------------------------------------------------------
        /**
         * @class Covariance
         * @brief Computes the covariance matrix from the solver's Hessian.
         *
         * - Computes marginal and joint covariances of state variables.
         * - Uses Cholesky factorization results from Gauss-Newton solvers.
         */
        class Covariance {
        public:
            using Ptr = std::shared_ptr<Covariance>;
            using ConstPtr = std::shared_ptr<const Covariance>;

            // -----------------------------------------------------------------------------
            /** @brief Constructs covariance estimation from a problem */
            explicit Covariance(slam::problem::Problem& problem);

            // -----------------------------------------------------------------------------
            /** @brief Constructs covariance estimation from a solver */
            explicit Covariance(slam::solver::GaussNewtonSolver& solver);

            virtual ~Covariance() = default;

            // -----------------------------------------------------------------------------
            /** @brief Queries covariance of a single state variable */
            Eigen::MatrixXd query(const slam::eval::StateVariableBase::ConstPtr& var) const;

            // -----------------------------------------------------------------------------
            /** @brief Queries covariance between two state variables */
            Eigen::MatrixXd query(const slam::eval::StateVariableBase::ConstPtr& rvar,
                                  const slam::eval::StateVariableBase::ConstPtr& cvar) const;

            // -----------------------------------------------------------------------------
            /** @brief Queries joint covariance of multiple variables */
            Eigen::MatrixXd query(const std::vector<slam::eval::StateVariableBase::ConstPtr>& vars) const;

            // -----------------------------------------------------------------------------
            /** @brief Queries block covariance between row and column variables */
            Eigen::MatrixXd query(const std::vector<slam::eval::StateVariableBase::ConstPtr>& rvars,
                                  const std::vector<slam::eval::StateVariableBase::ConstPtr>& cvars) const;

        private:
            // -----------------------------------------------------------------------------
            slam::problem::StateVector::ConstPtr state_vector_;

            // -----------------------------------------------------------------------------
            using SolverType = Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper>;
            std::shared_ptr<SolverType> hessian_solver_;
        };

    }  // namespace solver
}  // namespace slam
