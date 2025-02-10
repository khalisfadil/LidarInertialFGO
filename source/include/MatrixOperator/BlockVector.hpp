#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>
#include <Eigen/Core>

#include "source/include/MatrixOperator/BlockDimensionIndexing.hpp"

namespace slam {
    namespace blockmatrix {

        // -----------------------------------------------------------------------------
        /**
         * @class BlockVector
         * @brief Represents a single-column, block-row vector.
         * 
         * This class provides:
         * - **Block-wise storage and retrieval** for structured data.
         * - **Efficient indexing using `BlockDimensionIndexing`**.
         * - **Seamless conversion to Eigen dense vector format**.
         */
        class BlockVector {
            public:

                // -----------------------------------------------------------------------------
                /** @brief Default constructor. */
                BlockVector() = default;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor initializing block sizes.
                 * @param blockRowSizes Sizes of each block row.
                 */
                explicit BlockVector(const std::vector<unsigned int>& blockRowSizes);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor with block sizes and initial data.
                 * @param blockRowSizes Sizes of each block row.
                 * @param data Initial data vector (must match total block size).
                 * @throws std::invalid_argument if data size does not match expected size.
                 */
                BlockVector(const std::vector<unsigned int>& blockRowSizes, const Eigen::VectorXd& data);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Set internal data from an Eigen vector.
                 * @param v The input vector.
                 * @throws std::invalid_argument if size mismatch occurs.
                 */
                void setFromScalar(const Eigen::VectorXd& v);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Get indexing object for block structure.
                 * @return A constant reference to the `BlockDimensionIndexing` object.
                 */
                const BlockDimensionIndexing& getIndexing() const;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Adds a vector to a specific block entry.
                 * @param r Block index.
                 * @param v Vector to add.
                 * @throws std::out_of_range if block index is invalid.
                 * @throws std::invalid_argument if block size does not match.
                 */
                void add(unsigned int r, const Eigen::VectorXd& v);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves the block vector at a specific index.
                 * @param r Block index.
                 * @return The block vector as an Eigen::VectorXd.
                 * @throws std::out_of_range if block index is invalid.
                 */
                Eigen::VectorXd at(unsigned int r) const;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves a mapped block vector at a specific index.
                 * @param r Block index.
                 * @return A mapped Eigen::VectorXd reference to the data.
                 * @throws std::out_of_range if block index is invalid.
                 */
                Eigen::Map<Eigen::VectorXd> mapAt(unsigned int r);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Convert block vector to a dense Eigen vector.
                 * @return A reference to the internal Eigen vector.
                 */
                const Eigen::VectorXd& toEigen() const;

            private:
                BlockDimensionIndexing indexing_;  ///< Block indexing object.
                Eigen::VectorXd data_;  ///< Internal data storage.
        };

    }  // namespace blockmatrix
}  // namespace slam
