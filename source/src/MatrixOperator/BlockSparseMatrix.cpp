#include <include/MatrixOperator/BlockSparseMatrix.hpp>

namespace slam {

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

        cols_.clear();  // Ensure old values are erased
        cols_.resize(this->getIndexing().getColumnIndexing().getNumBlocksEntries());
    }

    // ----------------------------------------------------------------------------
    // Symmetric Block Matrix Constructor
    // ----------------------------------------------------------------------------

    BlockSparseMatrix::BlockSparseMatrix(const std::vector<unsigned int>& blockSizes, bool symmetric)
    : BlockMatrixBase(blockSizes, symmetric) {

        cols_.clear();  // Ensure no leftover data
        cols_.resize(this->getIndexing().getColumnIndexing().getNumBlocksEntries());
    }

    // ----------------------------------------------------------------------------
    // Clear Sparse Entries (TBB Optimized)
    // ----------------------------------------------------------------------------

    void BlockSparseMatrix::clear() {
        size_t numCols = this->getIndexing().getColumnIndexing().getNumBlocksEntries();
        if (numCols < 100) {  // Small matrices: use sequential loop
            for (size_t c = 0; c < numCols; c++) {
                cols_[c].rows.clear();
            }
        } else {  // Large matrices: use parallel execution
            tbb::parallel_for(size_t(0), numCols, [&](size_t c) {
                cols_[c].rows.clear();
            });
        }
    }

    // ----------------------------------------------------------------------------
    // Zero all matrix entries (TBB Optimized)
    // ----------------------------------------------------------------------------

    void BlockSparseMatrix::zero() {
        size_t numCols = this->getIndexing().getColumnIndexing().getNumBlocksEntries();
        
        if (numCols < 100) {  // Use sequential loop for small matrices
            for (size_t c = 0; c < numCols; c++) {
                for (auto& row_entry : cols_[c].rows) {
                    cols_[c].rows[row_entry.first].data.setZero();  // Access via key
                }
            }
        } else {  // Use parallel execution for large matrices
            tbb::parallel_for(size_t(0), numCols, [&](size_t c) {
                for (auto& row_entry : cols_[c].rows) {
                    cols_[c].rows[row_entry.first].data.setZero();  // Access via key
                }
            });
        }
    }

    // ----------------------------------------------------------------------------
    // Add Matrix to Sparse Block (TBB Optimized)
    // ----------------------------------------------------------------------------

    void BlockSparseMatrix::add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) {
        // Get references to indexing objects
        const BlockDimensionIndexing& blockRowIndexing = this->getIndexing().getRowIndexing();
        const BlockDimensionIndexing& blockColumnIndexing = this->getIndexing().getColumnIndexing();

        // Validate indices
        if (r >= blockRowIndexing.getNumBlocksEntries() || c >= blockColumnIndexing.getNumBlocksEntries()) {
            throw std::invalid_argument("[ERROR] Index (" + std::to_string(r) + ", " + std::to_string(c) + ") out of range.");
        }

        // Ensure upper-triangular portion is respected for symmetric matrices
        if (this->isScalarSymmetric() && r > c) {
            throw std::invalid_argument("[ERROR] Attempted to access lower half of an upper-symmetric block-sparse matrix.");
        }

        // Validate block dimensions
        if (m.rows() != static_cast<int>(blockRowIndexing.getBlockSizeAt(r)) ||
            m.cols() != static_cast<int>(blockColumnIndexing.getBlockSizeAt(c))) {
            throw std::invalid_argument("[ERROR] Block size mismatch at (" + std::to_string(r) + ", " + std::to_string(c) +
                                        "): Expected " + std::to_string(blockRowIndexing.getBlockSizeAt(r)) + "x" +
                                        std::to_string(blockColumnIndexing.getBlockSizeAt(c)) + ", got " +
                                        std::to_string(m.rows()) + "x" + std::to_string(m.cols()));
        }

        // **Guaranteed Mutable Reference Using `operator[]`**
        auto& entry = cols_[c].rows[r];

        // **Only initialize if uninitialized**
        if (entry.data.size() == 0) {
            entry.data = m;  // Direct assignment instead of setting Zero() first
        } else {
            entry.data += m;
        }
    }

    // ----------------------------------------------------------------------------
    // Get Row Entry at (r, c), Insert If Needed
    // ----------------------------------------------------------------------------

    BlockSparseMatrix::BlockRowEntry& BlockSparseMatrix::rowEntryAt(unsigned int r, unsigned int c, bool allowInsert) {
        // Validate indices
        const auto& blockRowIndexing = this->getIndexing().getRowIndexing();
        const auto& blockColumnIndexing = this->getIndexing().getColumnIndexing();

        if (r >= blockRowIndexing.getNumBlocksEntries() || c >= blockColumnIndexing.getNumBlocksEntries()) {
            throw std::invalid_argument("[ERROR] Index (" + std::to_string(r) + ", " + std::to_string(c) + ") out of range.");
        }

        // Ensure upper-triangular portion is respected for symmetric matrices
        if (this->isScalarSymmetric() && r > c) {
            throw std::invalid_argument("[ERROR] Attempted to access lower half of an upper-symmetric block-sparse matrix.");
        }

        // Get reference to the column
        auto& colRef = cols_[c].rows;

        // **Check existence before inserting**
        auto it = colRef.find(r);
        if (it != colRef.end()) {
            return const_cast<BlockRowEntry&>(it->second);  //  Ensures mutability
        }

        if (!allowInsert) {
            throw std::invalid_argument("[ERROR] Entry at (" + std::to_string(r) + ", " + std::to_string(c) + ") does not exist.");
        }

        // **Insert and retrieve the reference**
        auto& entry = colRef[r];
        entry.data = Eigen::MatrixXd::Zero(blockRowIndexing.getBlockSizeAt(r), blockColumnIndexing.getBlockSizeAt(c));

        return entry;
    }

    // ----------------------------------------------------------------------------
    // Get Matrix at (r, c)
    // ----------------------------------------------------------------------------

    Eigen::MatrixXd& BlockSparseMatrix::at(unsigned int r, unsigned int c) {
        // Validate indices
        const auto& blockRowIndexing = this->getIndexing().getRowIndexing();
        const auto& blockColumnIndexing = this->getIndexing().getColumnIndexing();

        if (r >= blockRowIndexing.getNumBlocksEntries() || c >= blockColumnIndexing.getNumBlocksEntries()) {
            throw std::invalid_argument("[ERROR] Index (" + std::to_string(r) + ", " + std::to_string(c) + ") out of range.");
        }

        // Ensure upper-triangular portion is respected for symmetric matrices
        if (this->isScalarSymmetric() && r > c) {
            throw std::invalid_argument("[ERROR] Attempted to access lower half of an upper-symmetric block-sparse matrix.");
        }

        // **Check existence first to avoid unnecessary insertion**
        auto& colRef = cols_[c].rows;
        auto it = colRef.find(r);
        if (it == colRef.end()) {
            throw std::invalid_argument("[ERROR] Entry at (" + std::to_string(r) + ", " + std::to_string(c) + ") does not exist.");
        }

        // **Fix: Remove `const` from `it->second` safely**
        return const_cast<Eigen::MatrixXd&>(it->second.data);
    }

    // ----------------------------------------------------------------------------
    // Get a Copy of Matrix at (r, c)
    // ----------------------------------------------------------------------------

    Eigen::MatrixXd BlockSparseMatrix::copyAt(unsigned int r, unsigned int c) const {
        // Validate indices
        const auto& blockRowIndexing = this->getIndexing().getRowIndexing();
        const auto& blockColumnIndexing = this->getIndexing().getColumnIndexing();

        if (r >= blockRowIndexing.getNumBlocksEntries() || c >= blockColumnIndexing.getNumBlocksEntries()) {
            throw std::invalid_argument("[ERROR] Index (" + std::to_string(r) + ", " + std::to_string(c) + ") out of range.");
        }

        // Reference to the correct column
        const auto& colRef = (this->isScalarSymmetric() && r > c) ? cols_[r] : cols_[c];

        // Find the entry
        auto it = colRef.rows.find((this->isScalarSymmetric() && r > c) ? c : r);
        if (it == colRef.rows.end()) {
            return Eigen::MatrixXd::Zero(blockRowIndexing.getBlockSizeAt(r), blockColumnIndexing.getBlockSizeAt(c));
        }

        return (this->isScalarSymmetric() && r > c) ? it->second.data.transpose() : it->second.data;
    }

    // ----------------------------------------------------------------------------
    // Convert to Eigen SparseMatrix (TBB Optimized)
    // ----------------------------------------------------------------------------

    Eigen::SparseMatrix<double> BlockSparseMatrix::toEigen(bool getSubBlockSparsity) const {
        // Get references to indexing objects
        const BlockDimensionIndexing& blockRowIndexing = this->getIndexing().getRowIndexing();
        const BlockDimensionIndexing& blockColumnIndexing = this->getIndexing().getColumnIndexing();

        // Allocate sparse matrix
        Eigen::SparseMatrix<double> mat(blockRowIndexing.getTotalScalarSize(), blockColumnIndexing.getTotalScalarSize());

        // **Use `tbb::concurrent_vector` for thread-safe storage**
        tbb::concurrent_vector<Eigen::Triplet<double>> tripletList;

        // **Parallel iteration over columns using TBB**
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blockColumnIndexing.getNumBlocksEntries()), [&](const tbb::blocked_range<size_t>& range) {
            tbb::concurrent_vector<Eigen::Triplet<double>> localTriplets;  // Per-thread triplets (reduces contention)

            for (size_t c = range.begin(); c < range.end(); ++c) {
                unsigned colBlkSize = blockColumnIndexing.getBlockSizeAt(c);
                unsigned colCumSum = blockColumnIndexing.getCumulativeBlockSizeAt(c);

                // Iterate over sparse row entries of the column
                for (const auto& [r, entry] : cols_[c].rows) {
                    unsigned rowBlkSize = blockRowIndexing.getBlockSizeAt(r);
                    unsigned rowCumSum = blockRowIndexing.getCumulativeBlockSizeAt(r);

                    // Iterate over internal matrix dimensions
                    unsigned colIdx = colCumSum;
                    for (unsigned j = 0; j < colBlkSize; j++, colIdx++) {
                        unsigned rowIdx = rowCumSum;
                        for (unsigned i = 0; i < rowBlkSize; i++, rowIdx++) {
                            double v_ij = entry.data(i, j);

                            // **Only insert non-zero values if `getSubBlockSparsity` is enabled**
                            if (v_ij != 0.0 || !getSubBlockSparsity) {
                                localTriplets.emplace_back(rowIdx, colIdx, v_ij);
                            }
                        }
                    }
                }
            }

            // **Corrected: Efficiently merge per-thread triplets using iterators**
            tripletList.grow_by(localTriplets.begin(), localTriplets.end());
        });

        // **Insert triplets into Eigen sparse matrix**
        mat.setFromTriplets(tripletList.begin(), tripletList.end());

        return mat;
    }

    // ----------------------------------------------------------------------------
    // Gets the number of non-zero entries per scalar-column
    // ----------------------------------------------------------------------------

    Eigen::VectorXi BlockSparseMatrix::getNnzPerCol() const {
        // Get references to indexing objects
        const BlockDimensionIndexing& blockColumnIndexing = this->getIndexing().getColumnIndexing();

        // Allocate vector of ints
        Eigen::VectorXi result(blockColumnIndexing.getTotalScalarSize());
        result.setZero();  // Initialize with zeros

        size_t numCols = blockColumnIndexing.getNumBlocksEntries();
        
        // Use sequential loop for small matrices (threshold is set at 100 columns here, can adjust)
        if (numCols < 100) {
            for (size_t c = 0; c < numCols; ++c) {
                unsigned nnz = 0;

                // Skip empty columns to avoid unnecessary iteration
                if (cols_[c].rows.empty()) continue;

                // Iterate over sparse row entries of the column
                for (const auto& [r, entry] : cols_[c].rows) {
                    nnz += entry.data.rows();
                }

                // Assign non-zero counts for the entire column block
                result.segment(blockColumnIndexing.getCumulativeBlockSizeAt(c), blockColumnIndexing.getBlockSizeAt(c)).setConstant(nnz);
            }
        } else {
            // Use parallel iteration for large matrices with TBB
            tbb::parallel_for(tbb::blocked_range<size_t>(0, numCols), [&](const tbb::blocked_range<size_t>& range) {
                for (size_t c = range.begin(); c < range.end(); ++c) {
                    unsigned nnz = 0;

                    // Skip empty columns to avoid unnecessary iteration
                    if (cols_[c].rows.empty()) continue;

                    // Iterate over sparse row entries of the column
                    for (const auto& [r, entry] : cols_[c].rows) {
                        nnz += entry.data.rows();
                    }

                    // Assign non-zero counts for the entire column block
                    result.segment(blockColumnIndexing.getCumulativeBlockSizeAt(c), blockColumnIndexing.getBlockSizeAt(c)).setConstant(nnz);
                }
            });
        }

        return result;
    }

} // namespace slam