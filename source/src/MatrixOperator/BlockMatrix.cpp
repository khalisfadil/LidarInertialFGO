#include "source/include/MatrixOperator/BlockMatrix.hpp"

namespace slam {
    namespace blockmatrix {

        // ----------------------------------------------------------------------------
        // Default Constructor
        // ----------------------------------------------------------------------------
        
        BlockMatrix::BlockMatrix() : BlockMatrixBase() {}

        // ----------------------------------------------------------------------------
        // Rectangular Block Matrix Constructor
        // ----------------------------------------------------------------------------
        
        BlockMatrix::BlockMatrix(const std::vector<unsigned int>& blockRowSizes,
                                const std::vector<unsigned int>& blockColumnSizes)
            : BlockMatrixBase(blockRowSizes, blockColumnSizes) {

            // Setup data structures
            data_.resize(this->getIndexing().getRowIndexing().getNumBlocksEntries());
            for (unsigned int r = 0; r < data_.size(); r++) {
                data_[r].resize(this->getIndexing().getColumnIndexing().getNumBlocksEntries());
            }
            this->zero();
        }

        // ----------------------------------------------------------------------------
        // Symmetric Block Matrix Constructor
        // ----------------------------------------------------------------------------

        BlockMatrix::BlockMatrix(const std::vector<unsigned int>& blockSizes, bool symmetric)
            : BlockMatrixBase(blockSizes, symmetric) {

            // Setup data structures
            data_.resize(this->getIndexing().getRowIndexing().getNumBlocksEntries());
            for (unsigned int r = 0; r < data_.size(); r++) {
                data_[r].resize(this->getIndexing().getColumnIndexing().getNumBlocksEntries());
            }
            this->zero();
        }

        // ----------------------------------------------------------------------------
        // Zeroing All Entries (Parallelized with TBB)
        // ----------------------------------------------------------------------------
        
        void BlockMatrix::zero() {
            tbb::parallel_for(tbb::blocked_range2d<size_t>(0, data_.size(), 0, data_[0].size()),
                [&](const tbb::blocked_range2d<size_t>& range) {
                    for (size_t r = range.rows().begin(); r < range.rows().end(); ++r) {
                        for (size_t c = range.cols().begin(); c < range.cols().end(); ++c) {
                            data_[r][c] = Eigen::MatrixXd::Zero(
                                this->getIndexing().getRowIndexing().getBlockSizeAt(r),
                                this->getIndexing().getColumnIndexing().getBlockSizeAt(c));
                        }
                    }
                });
        }

        // ----------------------------------------------------------------------------
        // Adding Matrices (Parallelized with TBB)
        // ----------------------------------------------------------------------------

        void BlockMatrix::add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) {
            // Get indexing objects
            const BlockDimensionIndexing& blockRowIndexing = this->getIndexing().getRowIndexing();
            const BlockDimensionIndexing& blockColumnIndexing = this->getIndexing().getColumnIndexing();

            // Check bounds
            if (r >= blockRowIndexing.getNumBlocksEntries() || c >= blockColumnIndexing.getNumBlocksEntries()) {
                throw std::invalid_argument("[BlockMatrix::add] Index (" + std::to_string(r) + ", " + std::to_string(c) +
                                            ") out of range. Row index must be less than " +
                                            std::to_string(blockRowIndexing.getNumBlocksEntries()) + " and column index must be less than " +
                                            std::to_string(blockColumnIndexing.getNumBlocksEntries()) + ".");
            }

            // Ensure upper-triangular portion is respected for symmetric matrices
            if (this->isScalarSymmetric() && r > c) {
                throw std::invalid_argument("[BlockMatrix::add] Attempted to access lower half of an upper-symmetric block matrix at index (" +
                                            std::to_string(r) + ", " + std::to_string(c) + ").");
            }

            // Check dimensions
            if (m.rows() != static_cast<int>(blockRowIndexing.getBlockSizeAt(r)) ||
                m.cols() != static_cast<int>(blockColumnIndexing.getBlockSizeAt(c))) {
                throw std::invalid_argument("[BlockMatrix::add] Matrix size mismatch at index (" + std::to_string(r) + ", " + std::to_string(c) +
                                            "). Expected size: (" + std::to_string(blockRowIndexing.getBlockSizeAt(r)) + ", " +
                                            std::to_string(blockColumnIndexing.getBlockSizeAt(c)) + "), but got size: (" +
                                            std::to_string(m.rows()) + ", " + std::to_string(m.cols()) + ").");
            }

            // Perform addition in parallel
            tbb::parallel_for(tbb::blocked_range<size_t>(0, m.rows()),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        data_[r][c].row(i) += m.row(i);
                    }
                });
        }

        // ----------------------------------------------------------------------------
        // Access Matrix at (r, c)
        // ----------------------------------------------------------------------------
        
        Eigen::MatrixXd& BlockMatrix::at(unsigned int r, unsigned int c) {
            // Bounds check
            if (r >= this->getIndexing().getRowIndexing().getNumBlocksEntries() ||
                c >= this->getIndexing().getColumnIndexing().getNumBlocksEntries()) {
                throw std::invalid_argument("[BlockMatrix::at] Index (" + std::to_string(r) + ", " + std::to_string(c) +
                                            ") out of range. Row index must be less than " + 
                                            std::to_string(this->getIndexing().getRowIndexing().getNumBlocksEntries()) +
                                            " and column index must be less than " +
                                            std::to_string(this->getIndexing().getColumnIndexing().getNumBlocksEntries()) + ".");
            }

            // Ensure upper-triangular portion is respected for symmetric matrices
            if (this->isScalarSymmetric() && r > c) {
                throw std::invalid_argument("[BlockMatrix::at] Attempted to access lower half of an upper-symmetric block matrix at index (" +
                                            std::to_string(r) + ", " + std::to_string(c) + ").");
            }

            return data_[r][c];
        }

        // ----------------------------------------------------------------------------
        // Copy Matrix at (r, c)
        // ----------------------------------------------------------------------------

        Eigen::MatrixXd BlockMatrix::copyAt(unsigned int r, unsigned int c) const {
            // Get indexing objects
            const BlockDimensionIndexing& blockRowIndexing = this->getIndexing().getRowIndexing();
            const BlockDimensionIndexing& blockColumnIndexing = this->getIndexing().getColumnIndexing();

            // Bounds check
            if (r >= blockRowIndexing.getNumBlocksEntries() || c >= blockColumnIndexing.getNumBlocksEntries()) {
                throw std::invalid_argument("[BlockMatrix::copyAt] Index (" + std::to_string(r) + ", " + std::to_string(c) +
                                            ") out of range in BlockMatrix. Row index must be less than " +
                                            std::to_string(blockRowIndexing.getNumBlocksEntries()) + " and column index must be less than " +
                                            std::to_string(blockColumnIndexing.getNumBlocksEntries()) + ".");
            }

            // Handle symmetric case
            if (this->isScalarSymmetric() && r > c) {
                return data_[c][r].transpose();  // Access upper triangular part
            }

            return data_[r][c];  // Return normal block
        }
    } // namespace blockmatrix
} // namespace slam