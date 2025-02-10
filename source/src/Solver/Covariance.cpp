#include "source/include/Solver/Covariance.hpp"
#include "source/include/MatrixOperator/BlockMatrix.hpp"
#include "source/include/MatrixOperator/BlockSparseMatrix.hpp"
#include "source/include/MatrixOperator/BlockVector.hpp"

namespace slam {
    namespace solver {

        // -----------------------------------------------------------------------------
        // Constructor: Builds covariance from the problem
        // -----------------------------------------------------------------------------

        Covariance::Covariance(slam::problem::Problem& problem)
            : state_vector_(problem.getStateVector()),
              hessian_solver_(std::make_shared<SolverType>()) {

            // Compute Hessian and factorize
            Eigen::SparseMatrix<double> approx_hessian;
            Eigen::VectorXd gradient_vector;
            problem.buildGaussNewtonTerms(approx_hessian, gradient_vector);
            hessian_solver_->analyzePattern(approx_hessian);
            hessian_solver_->factorize(approx_hessian);

            if (hessian_solver_->info() != Eigen::Success) {
                throw std::runtime_error("[Covariance::Covariance] Eigen LLT decomposition failed. Matrix may be ill-conditioned.");
            }
        }

        // -----------------------------------------------------------------------------
        // Constructor: Builds covariance from the solver
        // -----------------------------------------------------------------------------

        Covariance::Covariance(slam::solver::GaussNewtonSolver& solver)
            : state_vector_(solver.stateVector()), hessian_solver_(solver.getHessianSolver()) {
            if (!hessian_solver_) {
                throw std::runtime_error("[Covariance::Covariance] Solver Hessian decomposition is null.");
            }
        }

        // -----------------------------------------------------------------------------
        // Query: Single state variable
        // -----------------------------------------------------------------------------

        Eigen::MatrixXd Covariance::query(const slam::eval::StateVariableBase::ConstPtr& var) const {
            return query(std::vector<slam::eval::StateVariableBase::ConstPtr>{var});
        }

        // -----------------------------------------------------------------------------
        // Query: Covariance between two variables
        // -----------------------------------------------------------------------------

        Eigen::MatrixXd Covariance::query(
            const slam::eval::StateVariableBase::ConstPtr& rvar,
            const slam::eval::StateVariableBase::ConstPtr& cvar) const {
            return query({rvar}, {cvar});
        }

        // -----------------------------------------------------------------------------
        // Query: Joint covariance of multiple variables
        // -----------------------------------------------------------------------------

        Eigen::MatrixXd Covariance::query(
            const std::vector<slam::eval::StateVariableBase::ConstPtr>& vars) const {
            return query(vars, vars);
        }

        // -----------------------------------------------------------------------------
        // Query: Block covariance between row and column variables
        // -----------------------------------------------------------------------------

        Eigen::MatrixXd Covariance::query(
            const std::vector<slam::eval::StateVariableBase::ConstPtr>& rvars,
            const std::vector<slam::eval::StateVariableBase::ConstPtr>& cvars) const {
            
            auto state_vector = state_vector_;
            if (!state_vector) {
                throw std::runtime_error("[Covariance::query] State vector expired.");
            }

            // Create indexing
            slam::blockmatrix::BlockMatrixIndexing indexing(state_vector->getStateBlockSizes());
            const auto& blk_row_indexing = indexing.getRowIndexing();
            const auto& blk_col_indexing = indexing.getColumnIndexing();

            // Number of rows/cols
            const auto num_row_vars = rvars.size();
            const auto num_col_vars = cvars.size();

            // Block indices
            std::vector<unsigned int> blk_row_indices(num_row_vars);
            std::vector<unsigned int> blk_col_indices(num_col_vars);

            for (size_t i = 0; i < num_row_vars; i++) {
                blk_row_indices[i] = state_vector->getStateBlockIndex(rvars[i]->key());
            }
            for (size_t i = 0; i < num_col_vars; i++) {
                blk_col_indices[i] = state_vector->getStateBlockIndex(cvars[i]->key());
            }

            // Block sizes
            std::vector<unsigned int> blk_row_sizes(num_row_vars);
            std::vector<unsigned int> blk_col_sizes(num_col_vars);

            for (size_t i = 0; i < num_row_vars; i++) {
                blk_row_sizes[i] = blk_row_indexing.getBlockSizeAt(blk_row_indices[i]);
            }
            for (size_t i = 0; i < num_col_vars; i++) {
                blk_col_sizes[i] = blk_col_indexing.getBlockSizeAt(blk_col_indices[i]);
            }

            // Result container
            slam::blockmatrix::BlockMatrix cov_blk(blk_row_sizes, blk_col_sizes);
            const auto& cov_blk_indexing = cov_blk.getIndexing();

            // Compute covariance matrix
            for (unsigned int c = 0; c < num_col_vars; c++) {
                Eigen::VectorXd projection(blk_row_indexing.getTotalScalarSize());
                projection.setZero();

                for (unsigned int j = 0; j < blk_col_sizes[c]; j++) {
                    unsigned int scalar_col_index = blk_col_indexing.getCumulativeBlockSizeAt(blk_col_indices[c]) + j;
                    projection(scalar_col_index) = 1.0;

                    Eigen::VectorXd x = hessian_solver_->solve(projection);
                    projection(scalar_col_index) = 0.0;

                    for (unsigned int r = 0; r < num_row_vars; r++) {
                        int scalar_row_index = blk_row_indexing.getCumulativeBlockSizeAt(blk_row_indices[r]);
                        cov_blk.at(r, c).block(0, j, blk_row_sizes[r], 1) =
                            x.block(scalar_row_index, 0, blk_row_sizes[r], 1);
                    }
                }
            }

            // Convert to Eigen format
            Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(
                cov_blk_indexing.getRowIndexing().getTotalScalarSize(),
                cov_blk_indexing.getColumnIndexing().getTotalScalarSize()
            );

            for (unsigned int r = 0; r < num_row_vars; r++) {
                for (unsigned int c = 0; c < num_col_vars; c++) {
                    cov.block(cov_blk_indexing.getRowIndexing().getCumulativeBlockSizeAt(r),
                              cov_blk_indexing.getColumnIndexing().getCumulativeBlockSizeAt(c),
                              blk_row_sizes[r], blk_col_sizes[c]) = cov_blk.at(r, c);
                }
            }

            return cov;
        }

    }  // namespace solver
}  // namespace slam
