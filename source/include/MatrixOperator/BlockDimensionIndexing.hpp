#pragma once

#include <stdexcept>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace slam {

    /**
     * @class BlockDimensionIndexing
     * @brief Manages 1D indexing for block dimensions (rows or columns). 
     * 
     * Provides:
     * - Sizes of individual blocks (`blockSizes_`).
     * - Cumulative offsets for each block (`cumulativeBlockSizes_`).
     * - Total scalar size of all blocks (`totalScalarSize_`).
     * 
     * Acts as a fundamental component used by higher-level matrix indexing systems.
     */
    class BlockDimensionIndexing {

        public:

            // -----------------------------------------------------------------------------
            /**
             * @brief Default constructor. Initializes an empty indexing structure.
             */
            BlockDimensionIndexing();

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructor to initialize block sizes and compute cumulative sizes and total size.
             * 
             * @param blockSizes A vector of block sizes for rows or columns.
             * @throws std::invalid_argument if the input vector is empty or contains zero-sized blocks.
             */
            BlockDimensionIndexing(const std::vector<unsigned int>& blockSizes);

            // -----------------------------------------------------------------------------
            /**
             * @brief Returns the vector of block sizes.
             * 
             * @return A reference to the vector containing the size of each block.
             */
            const std::vector<unsigned int>& getBlockSizes() const; 

            // -----------------------------------------------------------------------------
            /**
             * @brief Returns the total number of blocks entries.
             * 
             * @return The number of blocks in the matrix.
             */
            unsigned int getNumBlocksEntries() const;

            // -----------------------------------------------------------------------------
            /**
             * @brief Returns the size of a block at the given index.
             * 
             * @param index The index of the block.
             * @return The size of the block at the specified index.
             * @throws std::out_of_range if the index is invalid.
             */
            unsigned int getBlockSizeAt(unsigned int index) const;

            // -----------------------------------------------------------------------------
            /**
             * @brief Returns the cumulative offset of a block at the given index.
             * 
             * @param index The index of the block.
             * @return The cumulative offset (starting position) of the block.
             * @throws std::out_of_range if the index is invalid.
             */
            unsigned int getCumulativeBlockSizeAt(unsigned int index) const;

            // -----------------------------------------------------------------------------
            /**
             * @brief Returns the total scalar size (sum of all block sizes).
             * 
             * @return The total scalar size of all blocks.
             */
            unsigned int getTotalScalarSize() const;

        private:
            // -----------------------------------------------------------------------------
            /**
             * @brief Stores the sizes of each block (e.g., row or column sizes).
             */
            std::vector<unsigned int> blockSizes_;

            // -----------------------------------------------------------------------------
            /**
             * @brief Stores the cumulative block sizes (offsets) for each block.
             */
            std::vector<unsigned int> cumulativeBlockSizes_;

            // -----------------------------------------------------------------------------
            /**
             * @brief Stores the total scalar size (sum of all block sizes).
             */
            unsigned int totalScalarSize_; //scalarDim_
    };
    
} // namespace slam
