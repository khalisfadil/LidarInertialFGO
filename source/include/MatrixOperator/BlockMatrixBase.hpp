#pragma once

#include <include/MatrixOperator/BlockDimensionIndexing.hpp>
#include <include/MatrixOperator/BlockMatrixIndexing.hpp>

#include <stdexcept>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace slam {
    namespace blockmatrix {

        // -----------------------------------------------------------------------------
        /**
         * @class BlockMatrixBase
         * @brief Base class for managing a block matrix structure.
         * 
         * - Supports **rectangular and symmetric** block matrices.
         * - Provides **block-wise access and manipulation**.
         * - Designed for **factor graph-based optimization**.
         */
        class BlockMatrixBase {
            
            public:
                // -----------------------------------------------------------------------------
                /** @brief Default constructor. */
                BlockMatrixBase() = default;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for symmetric block matrices (square).
                 * @param blockSizes Vector of block sizes.
                 * @param symmetric If true, assumes scalar-level symmetry.
                 */
                BlockMatrixBase(const std::vector<unsigned int>& blockSizes, bool symmetric);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for rectangular block matrices.
                 * @param blockRowSizes Vector of row block sizes.
                 * @param blockColumnSizes Vector of column block sizes.
                 */
                BlockMatrixBase(const std::vector<unsigned int>& blockRowSizes,
                                const std::vector<unsigned int>& blockColumnSizes);

                // -----------------------------------------------------------------------------
                /** @brief Virtual destructor (required for proper cleanup in derived classes). */
                virtual ~BlockMatrixBase() = default;

                // -----------------------------------------------------------------------------
                /** @brief Zero all matrix entries (must be implemented in derived class). */
                virtual void zero() = 0;

                // -----------------------------------------------------------------------------
                /** @brief Get the block matrix indexing structure. */
                const BlockMatrixIndexing& getIndexing() const { return indexing_; }

                // -----------------------------------------------------------------------------
                /** @brief Check if the matrix is symmetric at the scalar level. */
                bool isScalarSymmetric() const { return symmetric_; }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Adds a matrix to the block entry at (r, c).
                 * @param r Block row index.
                 * @param c Block column index.
                 * @param m Matrix to add (must match block size).
                 */
                virtual void add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) = 0;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Access a reference to the matrix at (r, c).
                 * @param r Block row index.
                 * @param c Block column index.
                 * @return Reference to the Eigen matrix.
                 */
                virtual Eigen::MatrixXd& at(unsigned int r, unsigned int c) = 0;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Get a copy of the matrix at (r, c).
                 * @param r Block row index.
                 * @param c Block column index.
                 * @return Copy of the Eigen matrix at (r, c).
                 */
                virtual Eigen::MatrixXd copyAt(unsigned int r, unsigned int c) const = 0;

            private:

                // -----------------------------------------------------------------------------
                /**
                 * @brief Helper function to initialize symmetric block indexing.
                 * @param blockSizes Vector of block sizes.
                 */
                void initializeSymmetricIndexing(const std::vector<unsigned int>& blockSizes);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Helper function to initialize rectangular block indexing.
                 * @param blockRowSizes Vector of row block sizes.
                 * @param blockColumnSizes Vector of column block sizes.
                 */
                void initializeRectangularIndexing(const std::vector<unsigned int>& blockRowSizes,
                                                const std::vector<unsigned int>& blockColumnSizes);

                // -----------------------------------------------------------------------------
                /**
                 * @brief True if the matrix is symmetric at the scalar level.
                 */
                bool symmetric_ = false;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Handles block-wise indexing.
                 */
                BlockMatrixIndexing indexing_;
        };
    } // namespace blockmatrix
} // namespace slam
