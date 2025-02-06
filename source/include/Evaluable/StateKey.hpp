#pragma once

#include <atomic>
#include <functional>

namespace slam {
    namespace eval {

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
         * @struct StateKeyHash
         * @brief Custom hash and equality functions for `StateKey`, compatible with TBB.
         */
        struct StateKeyHash {
            /**
             * @brief Hash function for `StateKey`.
             * @param key The state key to hash.
             * @return Hashed value.
             */
            static size_t hash(const StateKey& key) {
                return std::hash<unsigned int>{}(key);
            }

            /**
             * @brief Equality function for `StateKey`.
             * @param a First state key.
             * @param b Second state key.
             * @return `true` if keys are equal.
             */
            static bool equal(const StateKey& a, const StateKey& b) {
                return a == b;
            }
        };

        // -----------------------------------------------------------------------------
        /**
         * @brief Generates a new unique state key (thread-safe).
         *
         * Uses an atomic counter in relaxed mode for maximum performance
         * since only uniqueness is required (no ordering constraints).
         */
        inline StateKey NewStateKey() {
            static std::atomic<unsigned int> id{0};
            return id.fetch_add(1, std::memory_order_relaxed);
        }

    } // namespace eval
}  // namespace slam
