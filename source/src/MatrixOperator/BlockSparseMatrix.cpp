#include "source/include/MatrixOperator/BlockSparseMatrix.hpp"

namespace slam {
    namespace blockmatrix {

        // ----------------------------------------------------------------------------
        // Default Constructor
        // ----------------------------------------------------------------------------
        
        BlockSparseMatrix::BlockSparseMatrix() : BlockMatrixBase() {}

        // ----------------------------------------------------------------------------
        // Rectangular Block Matrix Constructor
        // ----------------------------------------------------------------------------

        BlockSparseMatrix::BlockSparseMatrix(const std::vector<unsigned int>& blockRowSizes,
                                             const std::vector<unsigned int>& blockColumnSizes)
            : BlockMatrixBase(blockRowSizes, blockColumnSizes) {
            cols_.resize(this->getIndexing().getColumnIndexing().getNumBlocksEntries());
        }

        // ----------------------------------------------------------------------------
        // Symmetric Block Matrix Constructor
        // ----------------------------------------------------------------------------

        BlockSparseMatrix::BlockSparseMatrix(const std::vector<unsigned int>& blockSizes, bool symmetric)
            : BlockMatrixBase(blockSizes, symmetric) {
            cols_.resize(this->getIndexing().getColumnIndexing().getNumBlocksEntries());
        }

        // ----------------------------------------------------------------------------
        // Clear Sparse Entries (TBB Optimized)
        // ----------------------------------------------------------------------------

        void BlockSparseMatrix::clear() {
            tbb::parallel_for(size_t(0), cols_.size(), [&](size_t c) {
                cols_[c].rows.clear();
            });
        }

        // ----------------------------------------------------------------------------
        // Zero all matrix entries (TBB Optimized)
        // ----------------------------------------------------------------------------

        void BlockSparseMatrix::zero() {
            tbb::parallel_for(size_t(0), cols_.size(), [&](size_t c) {
                for (auto it = cols_[c].rows.begin(); it != cols_[c].rows.end(); ++it) {
                    it->second.data.setZero();
                }
            });
        }

        // ----------------------------------------------------------------------------
        // Add Matrix to Sparse Block (TBB Optimized)
        // ----------------------------------------------------------------------------

        void BlockSparseMatrix::add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) {
            auto& column = cols_[c].rows;
            BlockSparseColumn::row_map_t::accessor accessor;
            column.insert(accessor, r);  // Ensures thread-safe access
            accessor->second.data += m;
        }

        // ----------------------------------------------------------------------------
        // Get Row Entry at (r, c), Insert If Needed
        // ----------------------------------------------------------------------------

        BlockSparseMatrix::BlockRowEntry& BlockSparseMatrix::rowEntryAt(unsigned int r, unsigned int c, bool allowInsert) {
            auto& column = cols_[c].rows;
            BlockSparseColumn::row_map_t::accessor accessor;
            if (!allowInsert && !column.find(accessor, r)) {
                throw std::invalid_argument("[BlockSparseMatrix::rowEntryAt] Entry does not exist.");
            }
            column.insert(accessor, r);
            return accessor->second;
        }

        // ----------------------------------------------------------------------------
        // Get Matrix at (r, c)
        // ----------------------------------------------------------------------------

        Eigen::MatrixXd& BlockSparseMatrix::at(unsigned int r, unsigned int c) {
            BlockSparseColumn::row_map_t::accessor accessor;
            if (!cols_[c].rows.find(accessor, r)) {
                throw std::invalid_argument("[BlockSparseMatrix::at] Entry does not exist.");
            }
            return accessor->second.data;
        }

        // ----------------------------------------------------------------------------
        // Get a Copy of Matrix at (r, c)
        // ----------------------------------------------------------------------------

        Eigen::MatrixXd BlockSparseMatrix::copyAt(unsigned int r, unsigned int c) const {
            BlockSparseColumn::row_map_t::const_accessor accessor;
            if (!cols_[c].rows.find(accessor, r)) {
                return Eigen::MatrixXd::Zero(
                    this->getIndexing().getRowIndexing().getBlockSizeAt(r),
                    this->getIndexing().getColumnIndexing().getBlockSizeAt(c)
                );
            }
            return accessor->second.data;
        }

        // ----------------------------------------------------------------------------
        // Convert to Eigen SparseMatrix (TBB Optimized)
        // ----------------------------------------------------------------------------

        Eigen::SparseMatrix<double> BlockSparseMatrix::toEigen(bool getSubBlockSparsity) const {
            Eigen::SparseMatrix<double> mat(this->getIndexing().getRowIndexing().getTotalScalarSize(),
                                            this->getIndexing().getColumnIndexing().getTotalScalarSize());

            tbb::concurrent_vector<Eigen::Triplet<double>> tripletList;

            tbb::parallel_for(tbb::blocked_range<size_t>(0, cols_.size()), [&](const tbb::blocked_range<size_t>& range) {
                tbb::concurrent_vector<Eigen::Triplet<double>> localTriplets;
                for (size_t c = range.begin(); c < range.end(); ++c) {
                    for (const auto& row_entry : cols_[c].rows) {
                        for (int i = 0; i < row_entry.second.data.rows(); ++i) {
                            for (int j = 0; j < row_entry.second.data.cols(); ++j) {
                                double value = row_entry.second.data(i, j);
                                if (value != 0.0 || !getSubBlockSparsity) {
                                    localTriplets.emplace_back(i, j, value);
                                }
                            }
                        }
                    }
                }
                tripletList.grow_by(localTriplets.begin(), localTriplets.end());
            });

            mat.setFromTriplets(tripletList.begin(), tripletList.end());
            return mat;
        }

        // ----------------------------------------------------------------------------
        // Get Non-Zero Entries Per Column (TBB Optimized)
        // ----------------------------------------------------------------------------

        Eigen::VectorXi BlockSparseMatrix::getNnzPerCol() const {
            Eigen::VectorXi result(this->getIndexing().getColumnIndexing().getTotalScalarSize());
            result.setZero();

            tbb::parallel_for(size_t(0), cols_.size(), [&](size_t c) {
                unsigned nnz = 0;
                for (const auto& row_entry : cols_[c].rows) {
                    nnz += row_entry.second.data.rows();
                }
                result.segment(
                    this->getIndexing().getColumnIndexing().getCumulativeBlockSizeAt(c),
                    this->getIndexing().getColumnIndexing().getBlockSizeAt(c)
                ).setConstant(nnz);
            });

            return result;
        }

    } // namespace blockmatrix
} // namespace slam
