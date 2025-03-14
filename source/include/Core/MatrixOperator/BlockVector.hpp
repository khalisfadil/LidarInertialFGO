#pragma once

#include <vector>
#include <Eigen/Core>

#include "Core/MatrixOperator/BlockDimensionIndexing.hpp"

namespace slam {
    namespace blockmatrix {
        
        // ----------------------------------------------------------------------------
        /**
         * @class BlockVector
         * @brief Represents a single-column, block-row vector for structured data.
         *
         * Provides block-wise storage, retrieval, and conversion to Eigen dense vector format,
         * with efficient indexing via `BlockDimensionIndexing`.
         */
        class BlockVector {
        public:

            // ----------------------------------------------------------------------------
            /** @brief Default constructor. */
            BlockVector() = default;

            // ----------------------------------------------------------------------------
            /**
             * @brief Constructs a block vector with specified block sizes, initialized to zero.
             * @param blockRowSizes Sizes of each block row.
             * @throws std::invalid_argument If blockRowSizes is empty.
             */
            explicit BlockVector(const std::vector<unsigned int>& blockRowSizes);

            // ----------------------------------------------------------------------------
            /**
             * @brief Constructs a block vector with block sizes and initial data.
             * @param blockRowSizes Sizes of each block row.
             * @param data Initial data vector (must match total block size).
             * @throws std::invalid_argument If blockRowSizes is empty or data size mismatches.
             */
            BlockVector(const std::vector<unsigned int>& blockRowSizes, const Eigen::VectorXd& data);

            // ----------------------------------------------------------------------------
            /**
             * @brief Sets internal data from an Eigen vector.
             * @param v Input vector (size must match total block size).
             * @throws std::invalid_argument If size mismatch occurs.
             */
            void setFromScalar(const Eigen::VectorXd& v);

            /** @brief Returns the block indexing structure. */
            const BlockDimensionIndexing& getIndexing() const noexcept { return indexing_; }

            // ----------------------------------------------------------------------------
            /**
             * @brief Adds a vector to the block at index r.
             * @param r Block index.
             * @param v Vector to add (size must match block size).
             * @throws std::out_of_range If r is invalid.
             * @throws std::invalid_argument If v size mismatches block size.
             */
            void add(unsigned int r, const Eigen::VectorXd& v);

            // ----------------------------------------------------------------------------
            /**
             * @brief Retrieves a copy of the block vector at index r.
             * @param r Block index.
             * @return Block vector as an Eigen::VectorXd.
             * @throws std::out_of_range If r is invalid.
             */
            Eigen::VectorXd at(unsigned int r) const;

            // ----------------------------------------------------------------------------
            /**
             * @brief Provides mapped access to the block vector at index r.
             * @param r Block index.
             * @return Mapped Eigen::VectorXd reference to the block data.
             * @throws std::out_of_range If r is invalid.
             */
            Eigen::Map<Eigen::VectorXd> mapAt(unsigned int r);

            // ----------------------------------------------------------------------------
            /** @brief Returns the entire block vector as a dense Eigen vector. */
            const Eigen::VectorXd& toEigen() const noexcept { return data_; }

        private:
            BlockDimensionIndexing indexing_;  ///< Manages block structure indexing.
            Eigen::VectorXd data_;             ///< Internal storage for vector data.
        };

    }  // namespace blockmatrix
}  // namespace slam