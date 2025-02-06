#pragma once

#include <tbb/concurrent_hash_map.h>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <stdexcept>
#include <iostream>

#include "source/include/MatrixOperator/BlockMatrixBase.hpp"

namespace slam {
    namespace blockmatrix {

        // -----------------------------------------------------------------------------
        /**
         * @class BlockSparseMatrix
         * @brief A block-sparse matrix optimized for real-time performance using Intel TBB.
         *
         * This class represents a block-sparse matrix where individual blocks are stored 
         * in a concurrent hash map for **efficient multi-threaded access**.
         * - Uses **Intel TBB (`tbb::concurrent_hash_map`)** for **thread-safe** block storage.
         * - Supports **parallelized operations** such as clearing, zeroing, and conversion.
         * - Efficiently handles **non-square and symmetric block structures**.
         */
        class BlockSparseMatrix : public BlockMatrixBase {
            
            private:
                // -----------------------------------------------------------------------------
                /** 
                 * @brief Computes the number of **non-zero entries** per scalar column.
                 *
                 * This function calculates the number of **active entries** per scalar column 
                 * in the block matrix, which is useful for optimizing sparse matrix storage 
                 * and conversions.
                 *
                 * @return Eigen::VectorXi - A vector containing the non-zero counts per column.
                 */
                Eigen::VectorXi getNnzPerCol() const;

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Represents a **single row entry** in the block-sparse matrix.
                 *
                 * Each row entry consists of an **Eigen dense block** (`Eigen::MatrixXd`) 
                 * that stores the actual numerical values of the block.
                 */
                struct BlockRowEntry {
                    Eigen::MatrixXd data;  ///< Dense block storage for matrix values.
                };

                // -----------------------------------------------------------------------------
                /**
                 * @brief Structure representing a **single sparse column** in the matrix.
                 *
                 * Each column stores a **hash map of row entries**, allowing for **fast lookups**,
                 * insertions, and modifications in a **thread-safe manner**.
                 * 
                 * - Uses `tbb::concurrent_hash_map` for **lock-free** row indexing.
                 * - Stores **only non-zero blocks**, reducing memory usage.
                 */
                struct BlockSparseColumn {
                    using row_map_t = tbb::concurrent_hash_map<unsigned int, BlockRowEntry>;
                    row_map_t rows;  ///< Thread-safe hash map storing row entries.
                };

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Storage for all **sparse columns** in the matrix.
                 *
                 * The matrix is stored **column-wise**, with each column containing 
                 * its own thread-safe `tbb::concurrent_hash_map` for **efficient sparse indexing**.
                 */
                std::vector<BlockSparseColumn> cols_;

            public:
                // -----------------------------------------------------------------------------
                /** 
                 * @brief Default constructor - Creates an **empty** block-sparse matrix.
                 *
                 * The matrix size must be **explicitly set** before usage.
                 */
                BlockSparseMatrix();

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Constructs a **rectangular block matrix** with specified block sizes.
                 *
                 * @param blockRowSizes Vector containing the sizes of each row block.
                 * @param blockColumnSizes Vector containing the sizes of each column block.
                 */
                BlockSparseMatrix(const std::vector<unsigned int>& blockRowSizes,
                                  const std::vector<unsigned int>& blockColumnSizes);

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Constructs a **symmetric block matrix** with specified block sizes.
                 *
                 * @param blockSizes Vector containing the sizes of each block.
                 * @param symmetric If `true`, enforces symmetry by storing only **upper-triangular** entries.
                 */
                BlockSparseMatrix(const std::vector<unsigned int>& blockSizes, bool symmetric = false);

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Clears **all sparse entries** while maintaining the matrix size.
                 *
                 * This operation is **parallelized using TBB**, ensuring **fast execution** 
                 * by clearing columns in parallel.
                 */
                void clear();

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Sets all **existing** matrix entries to zero.
                 *
                 * Unlike `clear()`, this function does **not remove** entries—it simply 
                 * sets their numerical values to zero, preserving the **sparsity structure**.
                 * 
                 * This operation is **TBB optimized** for **parallel execution**.
                 */
                void zero() override;

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Adds a **matrix block** at the specified position (r, c).
                 *
                 * If the entry at `(r, c)` does not exist, it will be **created automatically**.
                 * If it exists, the new values will be **added** to the existing block.
                 *
                 * @param r Row index (block level).
                 * @param c Column index (block level).
                 * @param m The Eigen matrix to be added.
                 *
                 * @throws std::invalid_argument If block sizes **do not match**.
                 */
                void add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) override;

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Returns a **reference** to the row entry at `(r, c)`, creating it if necessary.
                 *
                 * This function allows **direct modification** of block entries.
                 * If `allowInsert = false`, it throws an exception if the entry does not exist.
                 *
                 * @param r Row index (block level).
                 * @param c Column index (block level).
                 * @param allowInsert If `true`, inserts a new entry if it does not exist.
                 *
                 * @return BlockRowEntry& - A reference to the block at `(r, c)`.
                 * @throws std::invalid_argument If the entry does not exist and `allowInsert = false`.
                 */
                BlockRowEntry& rowEntryAt(unsigned int r, unsigned int c, bool allowInsert = false);

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Returns a **mutable reference** to the matrix at `(r, c)`.
                 *
                 * Unlike `rowEntryAt()`, this function does **not** insert a new block if it is missing.
                 *
                 * @param r Row index (block level).
                 * @param c Column index (block level).
                 *
                 * @return Eigen::MatrixXd& - Reference to the block matrix at `(r, c)`.
                 * @throws std::invalid_argument If the block does not exist.
                 */
                Eigen::MatrixXd& at(unsigned int r, unsigned int c) override;

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Returns a **copy** of the matrix at `(r, c)`.
                 *
                 * If the block does **not exist**, it returns a **zero-matrix** of appropriate size.
                 *
                 * @param r Row index (block level).
                 * @param c Column index (block level).
                 *
                 * @return Eigen::MatrixXd - Copy of the block at `(r, c)`, or a zero matrix if missing.
                 */
                Eigen::MatrixXd copyAt(unsigned int r, unsigned int c) const override;

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Converts the **block-sparse matrix** to an **Eigen sparse format**.
                 *
                 * This operation is **TBB optimized** using parallelized conversion:
                 * - Uses **Eigen::Triplet** format for efficient sparse storage.
                 * - Parallelized to accelerate **large matrix** conversions.
                 *
                 * @param getSubBlockSparsity If `true`, retains sub-block sparsity instead of merging.
                 * 
                 * @return Eigen::SparseMatrix<double> - The converted sparse matrix.
                 */
                Eigen::SparseMatrix<double> toEigen(bool getSubBlockSparsity = false) const;
        };

    } // namespace blockmatrix
} // namespace slam
