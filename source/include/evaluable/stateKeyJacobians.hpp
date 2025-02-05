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
     * @brief Manages and accumulates Jacobians associated with state keys in a TBB concurrent hash map.
     *
     * This class leverages TBB's concurrent_hash_map for safe concurrent insert/update.
     * It holds each state's Jacobian and accumulates it if multiple updates arrive.
     */
    class StateKeyJacobians {

        public:
            
            // -----------------------------------------------------------------------------
            /**
             * @brief Struct to hold the Jacobian and any related data.
             *        If per-key locking is needed, a std::mutex could be embedded here.
             */
            struct JacobianEntry {
                Eigen::MatrixXd mat;
            };

            using StateJacobianMap = tbb::concurrent_hash_map<StateKey, JacobianEntry, StateKeyHash>;

            StateKeyJacobians() = default;
            ~StateKeyJacobians() = default;

            // -----------------------------------------------------------------------------
            /**
             * @brief Inserts or accumulates a Jacobian for a given state key.
             *
             * - If the key is new, create a new entry.
             * - If it already exists, accumulate by adding the new Jacobian to the existing one.
             */
            void add(const StateKey& key, const Eigen::Ref<const Eigen::MatrixXd> jac) {
                typename StateJacobianMap::accessor acc;
                bool inserted = jacobian_map_.insert(acc, key);
                // If newly inserted, just store the value
                if (inserted) {
                    acc->second.mat = jac;
                } else {
                    // If it already existed, accumulate
                    acc->second.mat.noalias() += jac;
                }
                // Once acc goes out of scope, TBB releases the lock on that entry.
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Clears all stored Jacobians (not thread-safe if other threads are still adding).
             */
            void clear() {
                jacobian_map_.clear();
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves a copy of all stored (key, jacobian) pairs.
             *        Might not be a fully consistent snapshot if other threads modify the map concurrently.
             */
            std::vector<std::pair<StateKey, Eigen::MatrixXd>> getCopy() const {
                std::vector<std::pair<StateKey, Eigen::MatrixXd>> result;
                for (auto it = jacobian_map_.begin(); it != jacobian_map_.end(); ++it) {
                    result.emplace_back(it->first, it->second.mat);
                }
                return result;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Checks if a given state key exists in the Jacobian storage.
             */
            bool exists(const StateKey& key) const {
                return jacobian_map_.count(key) > 0;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves a Jacobian for a given state key (read-only).
             * @param key The key to look up.
             * @param jac Output matrix if found.
             * @return True if found, false otherwise.
             */
            bool get(const StateKey& key, Eigen::MatrixXd& jac) const {
                typename StateJacobianMap::const_accessor acc;
                if (jacobian_map_.find(acc, key)) {
                    jac = acc->second.mat;  // Copy to keep it thread-safe
                    return true;
                }
                return false;
            }

        private:
        
            StateJacobianMap jacobian_map_;
    };

}  // namespace slam
