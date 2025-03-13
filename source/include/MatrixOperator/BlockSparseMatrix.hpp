// Done
#pragma once

#include <tbb/concurrent_hash_map.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "MatrixOperator/BlockMatrixBase.hpp"

namespace slam {
    namespace blockmatrix {

        // -----------------------------------------------------------------------------
        /**
         * @class BlockSparseMatrix
         * @brief A thread-safe, block-sparse matrix optimized with Intel TBB for real-time performance.
         *
         * Stores non-zero blocks in a `tbb::concurrent_hash_map` for efficient, multi-threaded access.
         * Supports rectangular and symmetric structures with parallelized operations.
         */
        class BlockSparseMatrix : public BlockMatrixBase {
        private:   

            // -----------------------------------------------------------------------------
            /** @brief Represents a single dense block in the matrix. */
            struct BlockRowEntry {
                Eigen::MatrixXd data;

                BlockRowEntry() : data(Eigen::MatrixXd::Zero(0, 0)) {}
                BlockRowEntry(int rows, int cols) : data(Eigen::MatrixXd::Zero(rows, cols)) {}
            };

            // -----------------------------------------------------------------------------
            /** @brief Represents a sparse column with thread-safe row entries. */
            struct BlockSparseColumn {
                using row_map_t = tbb::concurrent_hash_map<unsigned int, BlockRowEntry>;
                row_map_t rows;
            };

            std::vector<BlockSparseColumn> cols_;  ///< Column-wise storage of sparse blocks.

            Eigen::VectorXi getNnzPerCol() const;  ///< Computes non-zero entries per scalar column.

        public:

            // -----------------------------------------------------------------------------
            /** @brief Constructs an empty block-sparse matrix (size must be set later). */
            BlockSparseMatrix() noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs a rectangular block-sparse matrix.
             * @param blockRowSizes Row block sizes.
             * @param blockColumnSizes Column block sizes.
             * @throws std::invalid_argument If sizes are empty.
             */
            BlockSparseMatrix(const std::vector<unsigned int>& blockRowSizes,
                            const std::vector<unsigned int>& blockColumnSizes);

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs a symmetric block-sparse matrix.
             * @param blockSizes Block sizes (square matrix).
             * @param symmetric If true, stores only upper-triangular entries.
             * @throws std::invalid_argument If sizes are empty.
             */
            BlockSparseMatrix(const std::vector<unsigned int>& blockSizes, bool symmetric = false);

            // -----------------------------------------------------------------------------
            /** @brief Clears all sparse entries, preserving matrix size (TBB-optimized). */
            void clear() noexcept;

            // -----------------------------------------------------------------------------
            /** @brief Sets all existing entries to zero, keeping sparsity (TBB-optimized). */
            void zero() override;

            // -----------------------------------------------------------------------------
            /**
             * @brief Adds a matrix block at (r, c), creating it if absent (thread-safe).
             * @param r Row block index.
             * @param c Column block index.
             * @param m Matrix to add (must match block size).
             * @throws std::invalid_argument If sizes mismatch.
             */
            void add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) override;

            // -----------------------------------------------------------------------------
            /**
             * @brief Accesses or inserts a block at (r, c).
             * @param r Row block index.
             * @param c Column block index.
             * @param allowInsert If true, creates block if absent; if false, throws if missing.
             * @return Reference to the block entry.
             * @throws std::out_of_range If indices are invalid.
             * @throws std::invalid_argument If block doesn’t exist and allowInsert is false.
             */
            BlockRowEntry& rowEntryAt(unsigned int r, unsigned int c, bool allowInsert = false);

            // -----------------------------------------------------------------------------
            /**
             * @brief Returns a mutable reference to the block at (r, c).
             * @param r Row block index.
             * @param c Column block index.
             * @return Reference to the block matrix.
             * @throws std::out_of_range If indices are invalid.
             * @throws std::invalid_argument If block doesn’t exist.
             */
            Eigen::MatrixXd& at(unsigned int r, unsigned int c) override;

            // -----------------------------------------------------------------------------
            /**
             * @brief Returns a copy of the block at (r, c), or zero matrix if absent.
             * @param r Row block index.
             * @param c Column block index.
             * @return Copy of the block matrix.
             * @throws std::out_of_range If indices are invalid.
             */
            Eigen::MatrixXd copyAt(unsigned int r, unsigned int c) const override;

            // -----------------------------------------------------------------------------
            /**
             * @brief Converts the block-sparse matrix to an Eigen sparse matrix (TBB-optimized).
             * @param getSubBlockSparsity If true, preserves sub-block sparsity.
             * @return Eigen::SparseMatrix<double> representation.
             */
            Eigen::SparseMatrix<double> toEigen(bool getSubBlockSparsity = false) const;
        };
    }  // namespace blockmatrix
}  // namespace slam