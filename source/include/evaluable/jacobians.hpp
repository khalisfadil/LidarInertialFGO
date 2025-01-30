#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <tbb/concurrent_hash_map.h>

#include <include/evaluable/stateKey.hpp>

namespace slam {
    
    // -----------------------------------------------------------------------------
    /**
     * @class StateKeyJacobians
     * @brief Manages and accumulates Jacobians associated with state keys in a concurrent hash map.
     * 
     * This class provides an efficient, thread-safe mechanism for storing and updating Jacobians 
     * related to different `StateKey` instances. It utilizes `tbb::concurrent_hash_map` to enable 
     * concurrent access while avoiding the need for explicit mutex locking.
     */
    class StateKeyJacobians {

        public:

            // -----------------------------------------------------------------------------
            /** 
             * @brief Type alias for a concurrent hash map that maps StateKeys to their corresponding Jacobians.
             * 
             * This hash map ensures efficient thread-safe storage and retrieval.
             */
            using StateJacobianMap = tbb::concurrent_hash_map<StateKey, Eigen::MatrixXd, StateKeyHash>;

            // -----------------------------------------------------------------------------
            /** 
             * @brief Default constructor. 
             * 
             * Initializes an empty storage for Jacobians.
             */
            StateKeyJacobians() = default;

            // -----------------------------------------------------------------------------
            /** 
             * @brief Default destructor. 
             * 
             * Ensures proper cleanup of allocated resources.
             */
            ~StateKeyJacobians() = default;
            
            // -----------------------------------------------------------------------------
            /**
             * @brief Inserts or updates a Jacobian for a given state key.
             * 
             * - If the `StateKey` does not exist in the map, a new entry is created with the provided Jacobian.
             * - If the `StateKey` already exists, the provided Jacobian is **accumulated (added)** to the existing one.
             * - This function is **thread-safe**, allowing concurrent insertions and updates.
             *
             * @param key The state key associated with the Jacobian.
             * @param jac The Jacobian matrix to be stored or accumulated.
             */
            void add(const StateKey &key, const Eigen::MatrixXd &jac) {
                typename StateJacobianMap::accessor acc;
                if (jacobian_map_.insert(acc, key)) {
                    // Key was newly inserted, set its value
                    acc->second = jac;
                } else {
                    // Key already exists, accumulate the Jacobian
                    acc->second += jac;
                }
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Clears all stored Jacobians.
             * 
             * This function removes all Jacobians from the storage, resetting it to an empty state.
             * It is useful for reinitialization or when the Jacobians need to be recomputed.
             */
            void clear() {
                jacobian_map_.clear();
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves a copy of all stored Jacobians.
             * 
             * - Since `tbb::concurrent_hash_map` does not support direct `const` access, 
             *   this function returns a **copy** of the internal map.
             * - The copy ensures that read operations do not interfere with ongoing writes.
             *
             * @return A copy of the internal concurrent hash map containing state keys and their associated Jacobians.
             */
            StateJacobianMap get_copy() const {
                return jacobian_map_; // Copying is required since concurrent_hash_map doesn’t support const access.
            }

        private:

            // -----------------------------------------------------------------------------
            /** 
             * @brief Concurrent hash map storing Jacobians mapped to their respective state keys.
             * 
             * This data structure allows for efficient parallel access and updates.
             */
            StateJacobianMap jacobian_map_;
    };

}  // namespace slam
