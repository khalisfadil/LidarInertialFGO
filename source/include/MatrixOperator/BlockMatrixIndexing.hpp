#pragma once

#include "source/include/MatrixOperator/BlockDimensionIndexing.hpp"

#include <stdexcept>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace slam {
    namespace blockmatrix {
        // -----------------------------------------------------------------------------
        /**
         * @class BlockMatrixIndexing
         * @brief Manages block indexing for rows and columns in a block matrix.
         * 
         * - Stores **block sizes** for both rows and columns.
         * - Computes **cumulative offsets** and **total scalar size**.
         * - Supports both **rectangular and symmetric** block matrices.
         */
        class BlockMatrixIndexing {

            public:

                // -----------------------------------------------------------------------------
                /**
                 * @brief Default constructor. Initializes an empty block matrix indexing structure.
                 */
                BlockMatrixIndexing();

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for a **symmetric block matrix**.
                 * 
                 * @param blockSizes A vector of block sizes (used for both rows and columns).
                 */
                BlockMatrixIndexing(const std::vector<unsigned int>& blockSizes);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for a **rectangular block matrix**.
                 * 
                 * @param blockRowSizes Block sizes for rows.
                 * @param blockColumnSizes Block sizes for columns.
                 */
                BlockMatrixIndexing(const std::vector<unsigned int>& blockRowSizes,
                                    const std::vector<unsigned int>& blockColumnSizes);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Returns the row indexing information.
                 * 
                 * @return Reference to `BlockDimensionIndexing` for rows.
                 */
                const BlockDimensionIndexing& getRowIndexing() const;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Returns the column indexing information.
                 * 
                 * @return Reference to `BlockDimensionIndexing` for columns.
                 */
                const BlockDimensionIndexing& getColumnIndexing() const;

            private:

                // -----------------------------------------------------------------------------
                /**
                 * @brief Stores row block sizes and indexing information.
                 */
                BlockDimensionIndexing blockRowIndexing_;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Stores column block sizes and indexing information.
                 */
                BlockDimensionIndexing blockColumnIndexing_;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Flag indicating whether the matrix is symmetric.
                 */
                bool blockSizeSymmetric_;
        };
    } // namespace blockmatrix
} // namespace slam
