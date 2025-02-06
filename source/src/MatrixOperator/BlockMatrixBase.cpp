#include "source/include/MatrixOperator/BlockMatrixBase.hpp"

namespace slam {
    namespace blockmatrix {

        // -----------------------------------------------------------------------------
        // Constructor: Symmetric (Square) Block Matrix
        // -----------------------------------------------------------------------------
        
        BlockMatrixBase::BlockMatrixBase(const std::vector<unsigned int>& blockSizes, bool symmetric)
        : symmetric_(symmetric) {
            if (blockSizes.empty()) {
                throw std::invalid_argument("[Error] Block matrix must have at least one block.");
            }
            initializeSymmetricIndexing(blockSizes);
        }

        // -----------------------------------------------------------------------------
        // Constructor: Rectangular Block Matrix
        // -----------------------------------------------------------------------------
        
        BlockMatrixBase::BlockMatrixBase(const std::vector<unsigned int>& blockRowSizes,
                                        const std::vector<unsigned int>& blockColumnSizes)
        : symmetric_(false) {
            if (blockRowSizes.empty() || blockColumnSizes.empty()) {
                throw std::invalid_argument("[Error] Rectangular block matrix must have row and column sizes.");
            }
            initializeRectangularIndexing(blockRowSizes, blockColumnSizes);
        }

        // -----------------------------------------------------------------------------
        // Helper Function: Initialize Indexing for Symmetric Matrices
        // -----------------------------------------------------------------------------
        
        void BlockMatrixBase::initializeSymmetricIndexing(const std::vector<unsigned int>& blockSizes) {
            indexing_ = std::move(BlockMatrixIndexing(blockSizes));
        }

        // -----------------------------------------------------------------------------
        // Helper Function: Initialize Indexing for Rectangular Matrices
        // -----------------------------------------------------------------------------
        
        void BlockMatrixBase::initializeRectangularIndexing(const std::vector<unsigned int>& blockRowSizes,
                                                            const std::vector<unsigned int>& blockColumnSizes) {
            indexing_ = std::move(BlockMatrixIndexing(blockRowSizes, blockColumnSizes));
        }

        // -----------------------------------------------------------------------------
        // Retrieve the block matrix indexing structure
        // -----------------------------------------------------------------------------
        
        const BlockMatrixIndexing& BlockMatrixBase::getIndexing() const {
            return indexing_;
        }

        // -----------------------------------------------------------------------------
        // Check if the block matrix is symmetric (scalar level)
        // -----------------------------------------------------------------------------
        
        bool BlockMatrixBase::isScalarSymmetric() const {
            return symmetric_;
        }
    } // namespace blockmatrix
} // namespace slam
