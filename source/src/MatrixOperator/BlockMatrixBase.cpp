#include "MatrixOperator/BlockMatrixBase.hpp"
#include <stdexcept>

namespace slam {
    namespace blockmatrix {

        // -----------------------------------------------------------------------------
        // Constructors
        // -----------------------------------------------------------------------------

        BlockMatrixBase::BlockMatrixBase(const std::vector<unsigned int>& blockSizes, bool symmetric)
            : symmetric_(symmetric) {
            if (blockSizes.empty()) {
                throw std::invalid_argument("[BlockMatrixBase] Block matrix must have at least one block.");
            }
            initializeSymmetricIndexing(blockSizes);
        }

        BlockMatrixBase::BlockMatrixBase(const std::vector<unsigned int>& blockRowSizes,
                                        const std::vector<unsigned int>& blockColumnSizes)
            : symmetric_(false) {
            if (blockRowSizes.empty() || blockColumnSizes.empty()) {
                throw std::invalid_argument("[BlockMatrixBase] Rectangular block matrix must have row and column sizes.");
            }
            initializeRectangularIndexing(blockRowSizes, blockColumnSizes);
        }

        // -----------------------------------------------------------------------------
        // Protected Helper Functions
        // -----------------------------------------------------------------------------

        void BlockMatrixBase::initializeSymmetricIndexing(const std::vector<unsigned int>& blockSizes) {
            indexing_ = BlockMatrixIndexing(blockSizes);  // Move assignment is implicit
        }

        void BlockMatrixBase::initializeRectangularIndexing(const std::vector<unsigned int>& blockRowSizes,
                                                            const std::vector<unsigned int>& blockColumnSizes) {
            indexing_ = BlockMatrixIndexing(blockRowSizes, blockColumnSizes);  // Move assignment is implicit
        }

    }  // namespace blockmatrix
}  // namespace slam