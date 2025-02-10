#include "source/include/MatrixOperator/BlockDimensionIndexing.hpp"

namespace slam {
    namespace blockmatrix {

        // -----------------------------------------------------------------------------
        // Section: BlockDimensionIndexing
        // -----------------------------------------------------------------------------
        
        BlockDimensionIndexing::BlockDimensionIndexing(){}

        // -----------------------------------------------------------------------------
        // Section: BlockDimensionIndexing
        // -----------------------------------------------------------------------------

        BlockDimensionIndexing::BlockDimensionIndexing(const std::vector<unsigned int>& blockSizes)
        : blockSizes_(blockSizes), totalScalarSize_(0) {

            // Check input validity
            if (blockSizes_.empty()) {
                throw std::invalid_argument("[BlockDimensionIndexing::BlockDimensionIndexing] Tried to initialize a block matrix with no size.");
            }

            // Initialize cumulative block sizes
            cumulativeBlockSizes_.reserve(blockSizes_.size());
            for (const auto& blockSize : blockSizes_) {
                // Check each block size is valid
                if (blockSize == 0) {
                    throw std::invalid_argument("[BlockDimensionIndexing::BlockDimensionIndexing] Tried to initialize a block row size of 0.");
                }

                // Add cumulative size
                cumulativeBlockSizes_.push_back(totalScalarSize_);
                totalScalarSize_ += blockSize;
            }
        }

        // -----------------------------------------------------------------------------
        // Section: getBlockSizes
        // -----------------------------------------------------------------------------
        
        const std::vector<unsigned int>& BlockDimensionIndexing::getBlockSizes() const {
            return blockSizes_;
        }

        // -----------------------------------------------------------------------------
        // Section: getBlockSizes
        // -----------------------------------------------------------------------------
        
        unsigned int BlockDimensionIndexing::getNumBlocksEntries() const {
            return blockSizes_.size();
        }

        // -----------------------------------------------------------------------------
        // Section: getBlockSizeAt
        // -----------------------------------------------------------------------------
        
        unsigned int BlockDimensionIndexing::getBlockSizeAt(unsigned int index) const {
            return blockSizes_.at(index);
        }

        // -----------------------------------------------------------------------------
        // Section: getCumulativeBlockSizeAt
        // -----------------------------------------------------------------------------
        
        unsigned int BlockDimensionIndexing::getCumulativeBlockSizeAt(unsigned int index) const {
            return cumulativeBlockSizes_.at(index);
        }

        // -----------------------------------------------------------------------------
        // Section: getCumulativeBlockSizeAt
        // -----------------------------------------------------------------------------
        
        unsigned int BlockDimensionIndexing::getTotalScalarSize() const {
            return totalScalarSize_;
        }
    } // namespace blockmatrix
} // namespace slam