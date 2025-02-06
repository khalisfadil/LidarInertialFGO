#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <tbb/concurrent_hash_map.h>

#include "source/include/Evaluable/StateKey.hpp"

namespace slam {
    namespace eval {

        // -----------------------------------------------------------------------------
        /**
         * @class StateKeyJacobians
         * @brief Thread-safe container for managing and accumulating Jacobians.
         *
         * This class maintains a **concurrent hash map** (`tbb::concurrent_hash_map`)  
         * that associates **state keys** with their corresponding **Jacobians**.  
         * 
         * **Functionality:**
         * - **Thread-safe insert/update:** Uses Intel TBB for **lock-free parallel processing**.
         * - **Automatic accumulation:** If a key exists, **accumulates the Jacobian** instead of replacing it.
         * - **Supports efficient lookup:** Allows **fast access** to stored Jacobians.
         * - **Snapshot retrieval:** Provides a **copy of all stored Jacobians**.
         *
         * **Use Case in SLAM:**
         * - Used in **factor graph optimizations** for **storing and retrieving linearized Jacobians**.
         * - Useful for **trajectory optimization, sensor fusion, and pose estimation**.
         */
        class StateKeyJacobians {

        public:

            // -----------------------------------------------------------------------------
            /**
             * @struct JacobianEntry
             * @brief Stores Jacobian matrices associated with a specific state key.
             */
            struct JacobianEntry {
                Eigen::MatrixXd mat;  ///< Jacobian matrix associated with the state.
            };

            /// **Concurrent Hash Map Definition**  
            using StateJacobianMap = tbb::concurrent_hash_map<StateKey, JacobianEntry, StateKeyHash>;

            // -----------------------------------------------------------------------------
            /**
             * @brief Default constructor.
             */
            StateKeyJacobians() = default;

            // -----------------------------------------------------------------------------
            /**
             * @brief Default destructor.
             */
            ~StateKeyJacobians() = default;

            // -----------------------------------------------------------------------------
            /**
             * @brief Inserts or accumulates a Jacobian for a given state key.
             *
             * - **If the key is new:** Creates a new entry.
             * - **If the key already exists:** Accumulates the new Jacobian with the existing one.
             *
             * @param key The state key to associate with the Jacobian.
             * @param jac The Jacobian matrix to be inserted or accumulated.
             */
            void add(const StateKey& key, const Eigen::Ref<const Eigen::MatrixXd> jac) {
                StateJacobianMap::accessor acc;
                bool inserted = jacobian_map_.insert(acc, key);
                
                if (inserted) {
                    // If newly inserted, store the Jacobian matrix.
                    acc->second.mat = jac;
                } else {
                    // If the key already exists, accumulate the Jacobian.
                    acc->second.mat.noalias() += jac;
                }
                // `acc` goes out of scope, releasing the lock automatically.
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Clears all stored Jacobians.
             *
             * **Note:** This function is **not thread-safe** if other threads  
             * are accessing the hash map while `clear()` is being called.
             */
            void clear() {
                jacobian_map_.clear();
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves a **copy** of all stored (key, Jacobian) pairs.
             *
             * @return A vector of key-Jacobian pairs.
             *
             * **Note:** This may not be a fully consistent snapshot if other  
             * threads modify the map while this function is running.
             */
            std::vector<std::pair<StateKey, Eigen::MatrixXd>> getCopy() const {
                std::vector<std::pair<StateKey, Eigen::MatrixXd>> result;
                for (const auto& it : jacobian_map_) {
                    result.emplace_back(it.first, it.second.mat);
                }
                return result;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Checks if a given state key exists in the Jacobian storage.
             * @param key The state key to check.
             * @return `true` if the key exists, otherwise `false`.
             */
            bool exists(const StateKey& key) const {
                return jacobian_map_.count(key) > 0;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves a Jacobian for a given state key.
             *
             * @param key The state key to look up.
             * @param[out] jac The output Jacobian matrix (if the key exists).
             * @return `true` if the Jacobian was found, otherwise `false`.
             */
            bool get(const StateKey& key, Eigen::MatrixXd& jac) const {
                StateJacobianMap::const_accessor acc;
                if (jacobian_map_.find(acc, key)) {
                    jac = acc->second.mat;  // Copying ensures thread safety.
                    return true;
                }
                return false;
            }

        private:
            StateJacobianMap jacobian_map_;  ///< Concurrent hash map storing state Jacobians.
        };

    } // namespace eval
}  // namespace slam
