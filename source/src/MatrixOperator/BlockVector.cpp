#include "MatrixOperator/BlockVector.hpp"
#include <stdexcept>
#include <string>

namespace slam {
    namespace blockmatrix {

        // -----------------------------------------------------------------------------
        // Constructors
        // -----------------------------------------------------------------------------

        BlockVector::BlockVector(const std::vector<unsigned int>& blockRowSizes)
            : indexing_(blockRowSizes), data_(Eigen::VectorXd::Zero(indexing_.getTotalScalarSize())) {
            if (blockRowSizes.empty()) {
                throw std::invalid_argument("[BlockVector] Block row sizes cannot be empty.");
            }
        }

        BlockVector::BlockVector(const std::vector<unsigned int>& blockRowSizes, const Eigen::VectorXd& v)
            : indexing_(blockRowSizes) {
            if (blockRowSizes.empty()) {
                throw std::invalid_argument("[BlockVector] Block row sizes cannot be empty.");
            }
            setFromScalar(v);
        }

        // -----------------------------------------------------------------------------
        // Data Manipulation
        // -----------------------------------------------------------------------------

        void BlockVector::setFromScalar(const Eigen::VectorXd& v) {
            if (indexing_.getTotalScalarSize() != static_cast<unsigned int>(v.size())) {
                throw std::invalid_argument("[BlockVector::setFromScalar] Size mismatch: expected " +
                                            std::to_string(indexing_.getTotalScalarSize()) + ", got " +
                                            std::to_string(v.size()));
            }
            data_ = v;
        }

        void BlockVector::add(unsigned int r, const Eigen::VectorXd& v) {
            if (r >= indexing_.getNumBlocksEntries()) {
                throw std::out_of_range("[BlockVector::add] Row index " + std::to_string(r) +
                                        " exceeds maximum " + std::to_string(indexing_.getNumBlocksEntries() - 1));
            }
            if (v.size() != static_cast<int>(indexing_.getBlockSizeAt(r))) {
                throw std::invalid_argument("[BlockVector::add] Block size mismatch at index " + std::to_string(r) +
                                            ": expected " + std::to_string(indexing_.getBlockSizeAt(r)) +
                                            ", got " + std::to_string(v.size()));
            }
            data_.segment(indexing_.getCumulativeBlockSizeAt(r), indexing_.getBlockSizeAt(r)) += v;
        }

        // -----------------------------------------------------------------------------
        // Data Access
        // -----------------------------------------------------------------------------

        Eigen::VectorXd BlockVector::at(unsigned int r) const {
            if (r >= indexing_.getNumBlocksEntries()) {
                throw std::out_of_range("[BlockVector::at] Row index " + std::to_string(r) +
                                        " exceeds maximum " + std::to_string(indexing_.getNumBlocksEntries() - 1));
            }
            return data_.segment(indexing_.getCumulativeBlockSizeAt(r), indexing_.getBlockSizeAt(r));
        }

        Eigen::Map<Eigen::VectorXd> BlockVector::mapAt(unsigned int r) {
            if (r >= indexing_.getNumBlocksEntries()) {
                throw std::out_of_range("[BlockVector::mapAt] Row index " + std::to_string(r) +
                                        " exceeds maximum " + std::to_string(indexing_.getNumBlocksEntries() - 1));
            }
            return Eigen::Map<Eigen::VectorXd>(data_.data() + indexing_.getCumulativeBlockSizeAt(r),
                                            indexing_.getBlockSizeAt(r));
        }

    }  // namespace blockmatrix
}  // namespace slam