#pragma once

#include <atomic>
#include <functional>

namespace slam {

    // -----------------------------------------------------------------------------
    /**
     * @file stateKey.hpp
     * @brief Provides a unique key system for identifying states in SLAM.
     */

    /**
     * @typedef StateKey
     * @brief Represents a unique state key as an unsigned integer.
     */
    using StateKey = unsigned int;

    // -----------------------------------------------------------------------------
    /**
     * @typedef StateKeyHash
     * @brief Hash function for StateKey.
     */
    using StateKeyHash = std::hash<unsigned int>;

    // -----------------------------------------------------------------------------
    /**
     * @brief Generates a new unique state key (thread-safe).
     *
     * Uses an atomic counter in relaxed mode for maximum performance
     * since only uniqueness is required (no ordering constraints).
     */
    inline StateKey NewStateKey() {
        static std::atomic<unsigned int> id{0};
        // No need for acq_rel if we only want a unique ID
        return id.fetch_add(1, std::memory_order_relaxed);
    }

}  // namespace slam
