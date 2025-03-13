#pragma once

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <Eigen/Core>
#include <vector>
#include <iostream>

#include "MatrixOperator/BlockMatrixBase.hpp"

namespace slam {
    namespace blockmatrix {

        // -----------------------------------------------------------------------------
        /**
         * @class BlockMatrix
         * @brief Implements a dense block matrix for factor graph-based optimization.
         * 
         * The `BlockMatrix` class provides a concrete implementation of a block matrix 
         * where each block is stored as an `Eigen::MatrixXd`. It supports operations such as:
         * - Initializing block matrices with given row and column sizes.
         * - Setting all entries to zero.
         * - Adding submatrices to specific block entries.
         * - Accessing and copying blocks.
         * 
         * Unlike `BlockMatrixBase`, this class stores actual data and provides 
         * full implementations for matrix operations.
         */
        class BlockMatrix : public BlockMatrixBase {

            public:

                // -----------------------------------------------------------------------------
                /** @brief Default constructor */
                BlockMatrix();

                // -----------------------------------------------------------------------------
                /** @brief Constructor for rectangular block matrices */
                BlockMatrix(const std::vector<unsigned int>& blockRowSizes,
                            const std::vector<unsigned int>& blockColumnSizes);

                // -----------------------------------------------------------------------------
                /** @brief Constructor for symmetric block matrices */
                BlockMatrix(const std::vector<unsigned int>& blockSizes, bool symmetric);

                // -----------------------------------------------------------------------------
                /** @brief Zero all matrix entries (TBB parallelized) */
                void zero() override;

                // -----------------------------------------------------------------------------
                /** @brief Add a matrix to the block entry at (r, c) (TBB parallelized) */
                void add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) override;
            
                // -----------------------------------------------------------------------------
                /** @brief Access a reference to the matrix at (r, c) */
                Eigen::MatrixXd& at(unsigned int r, unsigned int c) override;
            
                // -----------------------------------------------------------------------------
                /** @brief Get a copy of the matrix at (r, c) */
                Eigen::MatrixXd copyAt(unsigned int r, unsigned int c) const override;

            private:

                // -----------------------------------------------------------------------------
                /** @brief Stores the block matrices */
                std::vector<std::vector<Eigen::MatrixXd>> data_;
        };
    } // namespace blockmatrix
} // namespace slam

