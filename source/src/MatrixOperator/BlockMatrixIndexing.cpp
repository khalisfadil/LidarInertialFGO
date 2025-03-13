#include "MatrixOperator/BlockMatrixIndexing.hpp"

namespace slam {
    namespace blockmatrix {
        // -----------------------------------------------------------------------------
        // Default Constructor (Empty Block Matrix)
        // -----------------------------------------------------------------------------
        
        BlockMatrixIndexing::BlockMatrixIndexing()
            : blockRowIndexing_(), blockColumnIndexing_(), blockSizeSymmetric_(false) {}

        // -----------------------------------------------------------------------------
        // Constructor for Symmetric Block Matrix
        // -----------------------------------------------------------------------------
        
        BlockMatrixIndexing::BlockMatrixIndexing(const std::vector<unsigned int>& blockSizes)
            : blockRowIndexing_(blockSizes), 
            blockColumnIndexing_(blockSizes), // Ensure symmetry by initializing both
            blockSizeSymmetric_(true) {}

        // -----------------------------------------------------------------------------
        // Constructor for Rectangular Block Matrix
        // -----------------------------------------------------------------------------
        
        BlockMatrixIndexing::BlockMatrixIndexing(const std::vector<unsigned int>& blockRowSizes,
                                                const std::vector<unsigned int>& blockColumnSizes)
            : blockRowIndexing_(blockRowSizes), 
            blockColumnIndexing_(blockColumnSizes), 
            blockSizeSymmetric_(false) {}

        // -----------------------------------------------------------------------------
        // Get Row Indexing
        // -----------------------------------------------------------------------------
        
        const BlockDimensionIndexing& BlockMatrixIndexing::getRowIndexing() const {
            return blockRowIndexing_;
        }

        // -----------------------------------------------------------------------------
        // Get Column Indexing
        // -----------------------------------------------------------------------------
        
        const BlockDimensionIndexing& BlockMatrixIndexing::getColumnIndexing() const {
            if (blockSizeSymmetric_) {
                return blockRowIndexing_; // Use row indexing for columns
            }
            return blockColumnIndexing_;
        }
    } // namespace blockmatrix
} // namespace slam

