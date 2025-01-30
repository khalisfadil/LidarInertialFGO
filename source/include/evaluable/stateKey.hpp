#pragma once

#include <atomic>
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>             // UUID class
#include <boost/uuid/uuid_generators.hpp>  // Generators
#include <boost/uuid/uuid_io.hpp>          // Streaming operators

namespace slam {

    // -----------------------------------------------------------------------------
    /**
     * @file state_key.hpp
     * @brief Defines a unique state key system for tracking states in SLAM.
     * 
     * This file provides an abstraction for generating unique state keys, 
     * supporting either UUID-based or atomic integer-based keys. 
     * The key type is determined by the `USE_UUID` flag.
     */

    // -----------------------------------------------------------------------------
    /** 
     * @brief Compile-time flag to select the type of state key.
     * 
     * - `true` → Uses UUIDs (`boost::uuids::uuid`) for globally unique keys.
     * - `false` → Uses atomic unsigned integers (`unsigned int`) for fast key generation.
     */
    constexpr bool USE_UUID = false;

    // -----------------------------------------------------------------------------
    #if USE_UUID

        // -----------------------------------------------------------------------------
        /**
         * @typedef StateKey
         * @brief Type alias for a state key using UUIDs.
         * 
         * When `USE_UUID` is `true`, each state key is represented by a 
         * universally unique identifier (`boost::uuids::uuid`).
         */
        using StateKey = boost::uuids::uuid;

        // -----------------------------------------------------------------------------
        /**
         * @typedef StateKeyHash
         * @brief Hash function for `StateKey` when using UUIDs.
         * 
         * Uses `boost::hash<boost::uuids::uuid>` to enable UUID-based keys
         * in hash-based containers (e.g., `std::unordered_map`).
         */
        using StateKeyHash = boost::hash<boost::uuids::uuid>;

        // -----------------------------------------------------------------------------
        /**
         * @brief Generates a new unique state key using a random UUID generator.
         * 
         * - UUIDs ensure **global uniqueness** without requiring locks.
         * - Uses a **thread-local** random generator for efficiency in multithreading.
         * 
         * @return A randomly generated `StateKey` (UUID).
         */
        inline StateKey NewStateKey() {
            thread_local boost::uuids::random_generator uuid_gen;
            return uuid_gen();
        }

    #else  // USE_UUID == false

        // -----------------------------------------------------------------------------
        /**
         * @typedef StateKey
         * @brief Type alias for a state key using atomic unsigned integers.
         * 
         * When `USE_UUID` is `false`, each state key is represented by a fast-incrementing
         * `unsigned int`, providing efficient and lock-free key generation.
         */
        using StateKey = unsigned int;

        // -----------------------------------------------------------------------------
        /**
         * @typedef StateKeyHash
         * @brief Hash function for `StateKey` when using integers.
         * 
         * Uses `std::hash<unsigned int>` to enable integer-based keys in 
         * hash-based containers (e.g., `std::unordered_map`).
         */
        using StateKeyHash = std::hash<unsigned int>;

        // -----------------------------------------------------------------------------
        /**
         * @brief Generates a new unique state key using an atomic counter.
         * 
         * - This method ensures **fast, lock-free** key generation.
         * - It uses `std::atomic<unsigned int>` to guarantee **thread safety**.
         * - The `memory_order_relaxed` operation provides high performance 
         *   without unnecessary synchronization.
         * 
         * @return A new unique `StateKey` (unsigned integer).
         */
        inline StateKey NewStateKey() {
            static std::atomic<unsigned int> id = 0;
            return id.fetch_add(1, std::memory_order_relaxed);
        }

    #endif  // USE_UUID

}  // namespace slam
