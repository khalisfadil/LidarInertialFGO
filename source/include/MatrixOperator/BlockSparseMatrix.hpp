#pragma once

#include <include/MatrixOperator/BlockMatrixBase.hpp>
#include <3rdparty/robin-map/include/tsl/robin_map.h>  // Faster hash map alternative to std::map

#include <tbb/parallel_for.h>
#include <tbb/concurrent_hash_map.h>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include <stdexcept>
#include <iostream>

namespace slam {

    /**
     * @class BlockSparseMatrix
     * @brief A block-sparse matrix optimized using Intel TBB.
     *
     * This class implements a sparse block matrix using `tsl::robin_map` for fast hash-based
     * indexing and replaces OpenMP locks with Intel TBB's task-based parallelism.
     */
    class BlockSparseMatrix : public BlockMatrixBase {
        
        public:

            /** @brief Default constructor */
            BlockSparseMatrix();

            /** @brief Rectangular block matrix constructor */
            BlockSparseMatrix(const std::vector<unsigned int>& blockRowSizes,
                            const std::vector<unsigned int>& blockColumnSizes);

            /** @brief Block-size-symmetric matrix constructor */
            BlockSparseMatrix(const std::vector<unsigned int>& blockSizes, bool symmetric = false);

            /** @brief Clears all sparse entries but maintains matrix size */
            void clear();

            /** @brief Sets all matrix entries to zero (TBB optimized) */
            void zero() override;

            /** @brief Adds the matrix to the block entry at index (r, c), ensuring block dimensions match */
            void add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) override;

            /** 
             * @brief Returns a reference to the value at (r, c), if it exists.
             * @note Throws an exception if matrix is symmetric and a lower triangular entry is requested.
             */
            Eigen::MatrixXd& at(unsigned int r, unsigned int c) override;
        
            /** @brief Returns a copy of the matrix at index (r, c) */
            Eigen::MatrixXd copyAt(unsigned int r, unsigned int c) const override;
        
            /** @brief Converts to an Eigen sparse matrix */
            Eigen::SparseMatrix<double> toEigen(bool getSubBlockSparsity = false) const;
        
        private:

            /** @brief Computes the number of non-zero entries per scalar-column */
            Eigen::VectorXi getNnzPerCol() const;

            /** 
             * @brief Structure representing a row entry.
             * @note `tsl::robin_map` enables efficient parallel reads.
             */
            struct BlockRowEntry {
                Eigen::MatrixXd data;
            };

            /**
             * @brief Structure representing a sparse column.
             * Uses `tsl::robin_map` for efficient row indexing.
             */
            struct BlockSparseColumn {
                tsl::robin_map<unsigned int, BlockRowEntry> rows;  ///< Faster than `std::map`
            };

            /** @brief Storage for sparse columns */
            std::vector<BlockSparseColumn> cols_;

    };

} // namespace slam