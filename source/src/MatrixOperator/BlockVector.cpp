#include <stdexcept>
#include <sstream>

#include "source/include/MatrixOperator/BlockVector.hpp"

namespace slam {
    namespace blockmatrix {

        // ----------------------------------------------------------------------------
        // Default Constructor
        // ----------------------------------------------------------------------------
        
        BlockVector::BlockVector() = default;

        // ----------------------------------------------------------------------------
        // Block size constructor
        // ----------------------------------------------------------------------------

        BlockVector::BlockVector(const std::vector<unsigned int>& blkRowSizes)
            : indexing_(blkRowSizes), data_(Eigen::VectorXd::Zero(indexing_.getTotalScalarSize())) {}

        // ----------------------------------------------------------------------------
        // Block size (with data) constructor
        // ----------------------------------------------------------------------------

        BlockVector::BlockVector(const std::vector<unsigned int>& blkRowSizes, const Eigen::VectorXd& v)
            : indexing_(blkRowSizes) {
            setFromScalar(v);
        }

        // ----------------------------------------------------------------------------
        // Set internal data (ensures size consistency)
        // ----------------------------------------------------------------------------

        void BlockVector::setFromScalar(const Eigen::VectorXd& v) {
            if (indexing_.getTotalScalarSize() != static_cast<unsigned int>(v.size())) {
                throw std::invalid_argument("[BlockVector::setFromScalar] Size mismatch. Expected " +
                                            std::to_string(indexing_.getTotalScalarSize()) + ", but got " +
                                            std::to_string(v.size()) + ".");
            }
            data_ = v;
        }

        // ----------------------------------------------------------------------------
        // Get indexing object
        // ----------------------------------------------------------------------------

        const BlockDimensionIndexing& BlockVector::getIndexing() const {
            return indexing_;
        }

        // ----------------------------------------------------------------------------
        // Adds the vector to the block entry at index 'r'
        // ----------------------------------------------------------------------------

        void BlockVector::add(unsigned int r, const Eigen::VectorXd& v) {
            if (r >= indexing_.getNumBlocksEntries()) {
                throw std::out_of_range("[BlockVector::add] Row index " + std::to_string(r) +
                                        " out of range. Maximum allowed is " +
                                        std::to_string(indexing_.getNumBlocksEntries() - 1) + ".");
            }

            if (v.rows() != static_cast<int>(indexing_.getBlockSizeAt(r))) {
                throw std::invalid_argument("[BlockVector::add] Block size mismatch at index " +
                                            std::to_string(r) + ". Expected " +
                                            std::to_string(indexing_.getBlockSizeAt(r)) + ", but got " +
                                            std::to_string(v.rows()) + ".");
            }

            data_.segment(indexing_.getCumulativeBlockSizeAt(r), indexing_.getBlockSizeAt(r)) += v;
        }

        // ----------------------------------------------------------------------------
        // Return block vector at index 'r'
        // ----------------------------------------------------------------------------

        Eigen::VectorXd BlockVector::at(unsigned int r) const {
            if (r >= indexing_.getNumBlocksEntries()) {
                throw std::out_of_range("[BlockVector::at] Row index " + std::to_string(r) +
                                        " out of range in at().");
            }

            return data_.segment(indexing_.getCumulativeBlockSizeAt(r), indexing_.getBlockSizeAt(r));
        }

        // ----------------------------------------------------------------------------
        // Return mapped block vector at index 'r'
        // ----------------------------------------------------------------------------

        Eigen::Map<Eigen::VectorXd> BlockVector::mapAt(unsigned int r) {
            if (r >= indexing_.getNumBlocksEntries()) {
                throw std::out_of_range("[BlockVector::mapAt] Row index " + std::to_string(r) +
                                        " out of range in mapAt().");
            }

            return Eigen::Map<Eigen::VectorXd>(&data_[indexing_.getCumulativeBlockSizeAt(r)],
                                            indexing_.getBlockSizeAt(r), 1);
        }

        // ----------------------------------------------------------------------------
        // Convert to Eigen dense vector format
        // ----------------------------------------------------------------------------

        const Eigen::VectorXd& BlockVector::toEigen() const {
            return data_;
        }

    }  // namespace blockmatrix
}  // namespace slam
