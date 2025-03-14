#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include "Core/MatrixOperator/BlockDimensionIndexing.hpp"
#include "Core/MatrixOperator/BlockMatrixIndexing.hpp"

namespace slam {
    namespace blockmatrix {

        // -----------------------------------------------------------------------------
        /**
         * @class BlockMatrixBase
         * @brief Base class for managing block matrix structures in factor graph optimization.
         *
         * Supports both symmetric (square) and rectangular block matrices, providing block-wise access
         * and manipulation. Derived classes must implement specific matrix operations.
         */
        class BlockMatrixBase {
        public:

            // -----------------------------------------------------------------------------
            /** @brief Default constructor. */
            BlockMatrixBase() = default;

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs a symmetric (square) block matrix.
             * @param blockSizes Vector of block sizes (defines square matrix).
             * @param symmetric If true, assumes scalar-level symmetry.s
             * @throws std::invalid_argument If blockSizes is empty.
             */
            BlockMatrixBase(const std::vector<unsigned int>& blockSizes, bool symmetric);

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs a rectangular block matrix.
             * @param blockRowSizes Vector of row block sizes.
             * @param blockColumnSizes Vector of column block sizes.
             * @throws std::invalid_argument If either blockRowSizes or blockColumnSizes is empty.
             */
            BlockMatrixBase(const std::vector<unsigned int>& blockRowSizes,
                            const std::vector<unsigned int>& blockColumnSizes);

            // -----------------------------------------------------------------------------
            /** @brief Virtual destructor for proper cleanup in derived classes. */
            virtual ~BlockMatrixBase() noexcept = default;

            // -----------------------------------------------------------------------------
            /** @brief Zeros all matrix entries (must be implemented by derived classes). */
            virtual void zero() = 0;

            // -----------------------------------------------------------------------------
            /** @brief Returns the block matrix indexing structure. */
            const BlockMatrixIndexing& getIndexing() const noexcept { return indexing_; }

            // -----------------------------------------------------------------------------
            /** @brief Checks if the matrix is symmetric at the scalar level. */
            bool isScalarSymmetric() const noexcept { return symmetric_; }

            // -----------------------------------------------------------------------------
            /**
             * @brief Adds a matrix to the block entry at (r, c).
             * @param r Block row index.
             * @param c Block column index.
             * @param m Matrix to add (must match block size).
             */
            virtual void add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) = 0;

            // -----------------------------------------------------------------------------
            /**
             * @brief Accesses a reference to the matrix at (r, c).
             * @param r Block row index.
             * @param c Block column index.
             * @return Reference to the Eigen matrix.
             */
            virtual Eigen::MatrixXd& at(unsigned int r, unsigned int c) = 0;

            // -----------------------------------------------------------------------------
            /**
             * @brief Returns a copy of the matrix at (r, c).
             * @param r Block row index.
             * @param c Block column index.
             * @return Copy of the Eigen matrix.
             */
            virtual Eigen::MatrixXd copyAt(unsigned int r, unsigned int c) const = 0;

        protected:

            // -----------------------------------------------------------------------------
            /**
             * @brief Initializes indexing for a symmetric matrix.
             * @param blockSizes Vector of block sizes.
             */
            void initializeSymmetricIndexing(const std::vector<unsigned int>& blockSizes);

            // -----------------------------------------------------------------------------
            /**
             * @brief Initializes indexing for a rectangular matrix.
             * @param blockRowSizes Vector of row block sizes.
             * @param blockColumnSizes Vector of column block sizes.
             */
            void initializeRectangularIndexing(const std::vector<unsigned int>& blockRowSizes,
                                            const std::vector<unsigned int>& blockColumnSizes);

            bool symmetric_ = false;           ///< True if the matrix is symmetric at the scalar level.
            BlockMatrixIndexing indexing_;     ///< Manages block-wise indexing.
        };

    }  // namespace blockmatrix
}  // namespace slam