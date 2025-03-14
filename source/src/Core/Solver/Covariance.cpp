#include "Core/Solver/Covariance.hpp"
#include "Core/MatrixOperator/BlockMatrix.hpp"
#include "Core/MatrixOperator/BlockSparseMatrix.hpp"
#include "Core/MatrixOperator/BlockVector.hpp"

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

        Eigen::MatrixXd Covariance::query(const slam::eval::StateVariableBase::ConstPtr& rvar,
                                          const slam::eval::StateVariableBase::ConstPtr& cvar) const {
            return query(std::vector<slam::eval::StateVariableBase::ConstPtr>{rvar},
                         std::vector<slam::eval::StateVariableBase::ConstPtr>{cvar});
        }

        // -----------------------------------------------------------------------------
        // Query: Joint covariance of multiple variables
        // -----------------------------------------------------------------------------

        Eigen::MatrixXd Covariance::query(const std::vector<slam::eval::StateVariableBase::ConstPtr>& vars) const {
            return query(vars, vars);
        }

        // -----------------------------------------------------------------------------
        // Query: Block covariance between row and column variables
        // -----------------------------------------------------------------------------

        Eigen::MatrixXd Covariance::query(
            const std::vector<slam::eval::StateVariableBase::ConstPtr>& rvars,
            const std::vector<slam::eval::StateVariableBase::ConstPtr>& cvars) const {

            // Ensure state vector is valid
            auto state_vector = state_vector_.lock();
            if (!state_vector) {
                throw std::runtime_error("[Covariance::query] State vector expired.");
            }

            // Retrieve number of row/column variables
            const size_t num_row_vars = rvars.size();
            const size_t num_col_vars = cvars.size();

            // Return early if either dimension is empty
            if (num_row_vars == 0 || num_col_vars == 0) {
                return Eigen::MatrixXd::Zero(0, 0);
            }

            // Create indexing
            slam::blockmatrix::BlockMatrixIndexing indexing(state_vector->getStateBlockSizes());
            const auto& blk_row_indexing = indexing.getRowIndexing();
            const auto& blk_col_indexing = indexing.getColumnIndexing();

            // Reserve space for efficiency
            std::vector<unsigned int> blk_row_indices, blk_col_indices;
            std::vector<unsigned int> blk_row_sizes, blk_col_sizes;
            blk_row_indices.reserve(num_row_vars);
            blk_col_indices.reserve(num_col_vars);
            blk_row_sizes.reserve(num_row_vars);
            blk_col_sizes.reserve(num_col_vars);

            // Look up block indices and sizes
            for (const auto& rvar : rvars) {
                if (!rvar) throw std::runtime_error("[Covariance::query] Null row variable encountered.");
                blk_row_indices.emplace_back(state_vector->getStateBlockIndex(rvar->key()));
                blk_row_sizes.emplace_back(blk_row_indexing.getBlockSizeAt(blk_row_indices.back()));
            }
            for (const auto& cvar : cvars) {
                if (!cvar) throw std::runtime_error("[Covariance::query] Null column variable encountered.");
                blk_col_indices.emplace_back(state_vector->getStateBlockIndex(cvar->key()));
                blk_col_sizes.emplace_back(blk_col_indexing.getBlockSizeAt(blk_col_indices.back()));
            }

            // Create block matrix container
            slam::blockmatrix::BlockMatrix cov_blk(blk_row_sizes, blk_col_sizes);
            const auto& cov_blk_indexing = cov_blk.getIndexing();
            const auto& cov_blk_row_indexing = cov_blk_indexing.getRowIndexing();
            const auto& cov_blk_col_indexing = cov_blk_indexing.getColumnIndexing();

            // Pre-allocate projection vector
            Eigen::VectorXd projection = Eigen::VectorXd::Zero(blk_row_indexing.getTotalScalarSize());

            // Parallelized loop over columns using TBB
            tbb::parallel_for(tbb::blocked_range<size_t>(0, num_col_vars), [&](const tbb::blocked_range<size_t>& range) {
                for (size_t c = range.begin(); c < range.end(); ++c) {
                    const unsigned int scalar_col_index = blk_col_indexing.getCumulativeBlockSizeAt(blk_col_indices[c]);

                    for (unsigned int j = 0; j < blk_col_sizes[c]; ++j) {
                        projection[scalar_col_index + j] = 1.0;
                        Eigen::VectorXd x = hessian_solver_->solve(projection);
                        projection[scalar_col_index + j] = 0.0;

                        for (size_t r = 0; r < num_row_vars; ++r) {
                            cov_blk.at(r, c).col(j) = x.segment(
                                blk_row_indexing.getCumulativeBlockSizeAt(blk_row_indices[r]),
                                blk_row_sizes[r]
                            );
                        }
                    }
                }
            });

            // Convert block matrix to Eigen format
            Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(
                cov_blk_row_indexing.getTotalScalarSize(),
                cov_blk_col_indexing.getTotalScalarSize()
            );

            // Parallelizing final block conversion
            tbb::parallel_for(tbb::blocked_range<size_t>(0, num_row_vars), [&](const tbb::blocked_range<size_t>& row_range) {
                for (size_t r = row_range.begin(); r < row_range.end(); ++r) {
                    for (size_t c = 0; c < num_col_vars; ++c) {
                        cov.block(
                            cov_blk_row_indexing.getCumulativeBlockSizeAt(r),
                            cov_blk_col_indexing.getCumulativeBlockSizeAt(c),
                            blk_row_sizes[r], blk_col_sizes[c]
                        ) = cov_blk.at(r, c);
                    }
                }
            });

            return cov;
        }

    }  // namespace solver
}  // namespace slam