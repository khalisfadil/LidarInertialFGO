// Done
#include "Core/MatrixOperator/BlockSparseMatrix.hpp"
#include <stdexcept>

namespace slam {
    namespace blockmatrix {

        // -----------------------------------------------------------------------------
        // Constructors
        // -----------------------------------------------------------------------------

        BlockSparseMatrix::BlockSparseMatrix() noexcept : BlockMatrixBase() {}

        BlockSparseMatrix::BlockSparseMatrix(const std::vector<unsigned int>& blockRowSizes,
                                            const std::vector<unsigned int>& blockColumnSizes)
            : BlockMatrixBase(blockRowSizes, blockColumnSizes) {
            cols_.resize(getIndexing().getColumnIndexing().getNumBlocksEntries());
        }

        BlockSparseMatrix::BlockSparseMatrix(const std::vector<unsigned int>& blockSizes, bool symmetric)
            : BlockMatrixBase(blockSizes, symmetric) {
            cols_.resize(getIndexing().getColumnIndexing().getNumBlocksEntries());
        }

        // -----------------------------------------------------------------------------
        // Sparse Matrix Operations
        // -----------------------------------------------------------------------------

        void BlockSparseMatrix::clear() noexcept {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, cols_.size()),
                            [&](const tbb::blocked_range<size_t>& range) {
                                for (size_t c = range.begin(); c != range.end(); ++c) {
                                    cols_[c].rows.clear();
                                }
                            });
        }

        void BlockSparseMatrix::zero() {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, cols_.size()),
                            [&](const tbb::blocked_range<size_t>& range) {
                                for (size_t c = range.begin(); c != range.end(); ++c) {
                                    for (auto& row_entry : cols_[c].rows) {
                                        row_entry.second.data.setZero();
                                    }
                                }
                            });
        }

        void BlockSparseMatrix::add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) {
            if (r >= getIndexing().getRowIndexing().getNumBlocksEntries() ||
                c >= getIndexing().getColumnIndexing().getNumBlocksEntries()) {
                throw std::out_of_range("[BlockSparseMatrix::add] Index out of range: (" +
                                        std::to_string(r) + ", " + std::to_string(c) + ")");
            }
            if (m.rows() != getIndexing().getRowIndexing().getBlockSizeAt(r) ||
                m.cols() != getIndexing().getColumnIndexing().getBlockSizeAt(c)) {
                throw std::invalid_argument("[BlockSparseMatrix::add] Size mismatch at (" +
                                            std::to_string(r) + ", " + std::to_string(c) + ")");
            }

            BlockSparseColumn::row_map_t::accessor acc;
            auto& column = cols_[c].rows;
            if (column.emplace(acc, r, BlockRowEntry(m.rows(), m.cols()))) {  // New block
                acc->second.data = m;
            } else {  // Existing block
                acc->second.data += m;
            }
        }

        // -----------------------------------------------------------------------------
        // Block Access
        // -----------------------------------------------------------------------------

        BlockSparseMatrix::BlockRowEntry& BlockSparseMatrix::rowEntryAt(unsigned int r, unsigned int c, bool allowInsert) {
            if (c >= cols_.size()) {
                throw std::out_of_range("[BlockSparseMatrix::rowEntryAt] Column index " + std::to_string(c) +
                                        " exceeds maximum " + std::to_string(cols_.size() - 1));
            }
            auto& column = cols_[c].rows;
            BlockSparseColumn::row_map_t::accessor acc;
            if (!column.find(acc, r)) {
                if (!allowInsert) {
                    throw std::invalid_argument("[BlockSparseMatrix::rowEntryAt] Block (" +
                                                std::to_string(r) + ", " + std::to_string(c) + ") does not exist");
                }
                column.emplace(acc, r, BlockRowEntry(getIndexing().getRowIndexing().getBlockSizeAt(r),
                                                    getIndexing().getColumnIndexing().getBlockSizeAt(c)));
            }
            return acc->second;
        }

        Eigen::MatrixXd& BlockSparseMatrix::at(unsigned int r, unsigned int c) {
            if (c >= cols_.size()) {
                throw std::out_of_range("[BlockSparseMatrix::at] Column index " + std::to_string(c) +
                                        " exceeds maximum " + std::to_string(cols_.size() - 1));
            }
            BlockSparseColumn::row_map_t::accessor acc;
            if (!cols_[c].rows.find(acc, r)) {
                throw std::invalid_argument("[BlockSparseMatrix::at] Block (" +
                                            std::to_string(r) + ", " + std::to_string(c) + ") does not exist");
            }
            return acc->second.data;
        }

        Eigen::MatrixXd BlockSparseMatrix::copyAt(unsigned int r, unsigned int c) const {
            if (r >= getIndexing().getRowIndexing().getNumBlocksEntries() ||
                c >= getIndexing().getColumnIndexing().getNumBlocksEntries()) {
                throw std::out_of_range("[BlockSparseMatrix::copyAt] Index out of range: (" +
                                        std::to_string(r) + ", " + std::to_string(c) + ")");
            }
            BlockSparseColumn::row_map_t::const_accessor acc;
            if (!cols_[c].rows.find(acc, r)) {
                return Eigen::MatrixXd::Zero(getIndexing().getRowIndexing().getBlockSizeAt(r),
                                            getIndexing().getColumnIndexing().getBlockSizeAt(c));
            }
            return acc->second.data;
        }

        // -----------------------------------------------------------------------------
        // Conversion to Eigen Sparse Matrix
        // -----------------------------------------------------------------------------

        Eigen::SparseMatrix<double> BlockSparseMatrix::toEigen(bool getSubBlockSparsity) const {
            Eigen::SparseMatrix<double> mat(getIndexing().getRowIndexing().getTotalScalarSize(),
                                            getIndexing().getColumnIndexing().getTotalScalarSize());

            // Define a reduction body for collecting triplets
            struct TripletReducer {
                std::vector<Eigen::Triplet<double>> triplets;
                const BlockSparseMatrix& matrix;
                bool subBlockSparsity;

                // Constructor for initial thread
                TripletReducer(const BlockSparseMatrix& m, bool sparsity)
                    : matrix(m), subBlockSparsity(sparsity) {
                    triplets.reserve(1000);  // Pre-allocate for efficiency
                }

                // Splitting constructor for parallel tasks
                TripletReducer(TripletReducer& other, tbb::split)
                    : matrix(other.matrix), subBlockSparsity(other.subBlockSparsity) {
                    triplets.reserve(1000);
                }

                // Process a range of columns
                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t c = range.begin(); c != range.end(); ++c) {
                        const unsigned int colOffset = matrix.getIndexing().getColumnIndexing().getCumulativeBlockSizeAt(c);
                        for (const auto& row_entry : matrix.cols_[c].rows) {
                            const unsigned int r = row_entry.first;
                            const unsigned int rowOffset = matrix.getIndexing().getRowIndexing().getCumulativeBlockSizeAt(r);
                            const Eigen::MatrixXd& block = row_entry.second.data;
                            for (int i = 0; i < block.rows(); ++i) {
                                for (int j = 0; j < block.cols(); ++j) {
                                    double value = block(i, j);
                                    if (value != 0.0 || !subBlockSparsity) {
                                        triplets.emplace_back(rowOffset + i, colOffset + j, value);
                                    }
                                }
                            }
                        }
                    }
                }

                // Combine results from different threads
                void join(const TripletReducer& other) {
                    triplets.insert(triplets.end(), other.triplets.begin(), other.triplets.end());
                }
            };

            // Perform parallel reduction
            TripletReducer reducer(*this, getSubBlockSparsity);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, cols_.size()), reducer);

            // Build the sparse matrix from the combined triplets
            mat.setFromTriplets(reducer.triplets.begin(), reducer.triplets.end());
            return mat;
        }

        // -----------------------------------------------------------------------------
        // Non-Zero Entries Per Column
        // -----------------------------------------------------------------------------

        Eigen::VectorXi BlockSparseMatrix::getNnzPerCol() const {
            Eigen::VectorXi nnz(getIndexing().getColumnIndexing().getTotalScalarSize());
            nnz.setZero();

            tbb::parallel_for(tbb::blocked_range<size_t>(0, cols_.size()),
                            [&](const tbb::blocked_range<size_t>& range) {
                                for (size_t c = range.begin(); c != range.end(); ++c) {
                                    unsigned int count = 0;
                                    for (const auto& row_entry : cols_[c].rows) {
                                        count += row_entry.second.data.rows();
                                    }
                                    nnz.segment(getIndexing().getColumnIndexing().getCumulativeBlockSizeAt(c),
                                                getIndexing().getColumnIndexing().getBlockSizeAt(c))
                                        .setConstant(count);
                                }
                            });

            return nnz;
        }
    }  // namespace blockmatrix
}  // namespace slam