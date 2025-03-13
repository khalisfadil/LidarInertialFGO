#include <iostream>
#include <numeric>
#include <sstream>

#include "Problem/StateVector.hpp"
#include "MatrixOperator/BlockVector.hpp"

namespace slam {
    namespace problem {

        // ----------------------------------------------------------------------------
        // Explicit Copy Constructor
        // ----------------------------------------------------------------------------

        StateVector::StateVector(const StateVector& other)
            : states_(other.states_), num_block_entries_(other.num_block_entries_.load()) {
            // Perform deep copy of state variables
            for (auto& [key, entry] : states_) {
                entry.state = entry.state->clone();
            }
        }

        // ----------------------------------------------------------------------------
        // Clone Function (Uses Copy Constructor)
        // ----------------------------------------------------------------------------

        StateVector StateVector::clone() const {
            return StateVector(*this);
        }

        // ----------------------------------------------------------------------------
        // Copy Values from Another StateVector
        // ----------------------------------------------------------------------------

        void StateVector::copyValues(const StateVector& other) {
            if (states_.empty() ||
                num_block_entries_.load() != other.num_block_entries_.load() ||
                states_.size() != other.states_.size()) {
                throw std::invalid_argument("[StateVector::copyValues] size mismatch in copyValues()");
            }

            for (auto& [key, entry] : states_) {
                tbb::concurrent_hash_map<slam::eval::StateKey, StateContainer, slam::eval::StateKeyHashCompare>::const_accessor acc;
                
                if (!other.states_.find(acc, key)) {
                throw std::runtime_error("[StateVector::copyValues] structure mismatch in copyValues(): missing key " + std::to_string(key));
                }

                if (entry.local_block_index != acc->second.local_block_index) {
                throw std::runtime_error("[StateVector::copyValues] structure mismatch in copyValues(): index mismatch");
                }

                entry.state->setFromCopy(acc->second.state);
            }
        }

        // ----------------------------------------------------------------------------
        // Add a State Variable
        // ----------------------------------------------------------------------------

        void StateVector::addStateVariable(const slam::eval::StateVariableBase::Ptr& state) {
            if (state->locked()) {
                throw std::invalid_argument("[StateVector::getStateVariable] Cannot add a locked state variable to an optimizable StateVector.");
            }

            const auto& key = state->key();
            if (hasStateVariable(key)) {
                throw std::runtime_error("[StateVector::getStateVariable] StateVector already contains the given state.");
            }

            // Insert new state variable efficiently
            StateContainer new_entry{state, static_cast<int>(num_block_entries_++)};
            states_.emplace(key, std::move(new_entry));
        }

        // ----------------------------------------------------------------------------
        // Check if a State Variable Exists
        // ----------------------------------------------------------------------------

        bool StateVector::hasStateVariable(const slam::eval::StateKey& key) const noexcept {
            return states_.count(key) > 0;
        }

        // ----------------------------------------------------------------------------
        // Retrieve a State Variable by Key
        // ----------------------------------------------------------------------------

        slam::eval::StateVariableBase::ConstPtr StateVector::getStateVariable(
                const slam::eval::StateKey& key) const {
            
            tbb::concurrent_hash_map<slam::eval::StateKey, StateContainer, slam::eval::StateKeyHashCompare>::const_accessor acc;
            
            if (!states_.find(acc, key)) {
                throw std::runtime_error("[StateVector::getStateVariable] State variable not found in getStateVariable()");
            }

            return acc->second.state;
        }

        // ----------------------------------------------------------------------------
        // Get Number of States
        // ----------------------------------------------------------------------------
        

        unsigned int StateVector::getNumberOfStates() const noexcept {
            return states_.size();
        }

        // ----------------------------------------------------------------------------
        // Get Block Index of a Specific State
        // ----------------------------------------------------------------------------

        int StateVector::getStateBlockIndex(const slam::eval::StateKey& key) const {
        
            tbb::concurrent_hash_map<slam::eval::StateKey, StateContainer, slam::eval::StateKeyHashCompare>::const_accessor acc;
            
            if (!states_.find(acc, key)) {
                throw std::runtime_error("[StateVector::getStateBlockIndex] Requested state is missing in getStateBlockIndex()");
            }

            return acc->second.local_block_index;
        }

        // ----------------------------------------------------------------------------
        // Get Ordered List of Block Sizes
        // ----------------------------------------------------------------------------

        std::vector<unsigned int> StateVector::getStateBlockSizes() const {
            std::vector<unsigned int> result(states_.size());

            for (const auto& [key, entry] : states_) {
                if (entry.local_block_index < 0 || entry.local_block_index >= static_cast<int>(result.size())) {
                throw std::logic_error("[StateVector::getStateBlockSizes] Invalid local_block_index in getStateBlockSizes()");
                }
                result[entry.local_block_index] = entry.state->perturb_dim();
            }

            return result;
        }

        // ----------------------------------------------------------------------------
        // Get Total Size of the State Vector
        // ----------------------------------------------------------------------------

        unsigned int StateVector::getStateSize() const noexcept {
            return std::accumulate(states_.begin(), states_.end(), 0U,
                                    [](unsigned int sum, const auto& pair) {
                                    return sum + pair.second.state->perturb_dim();
                                    });
        }

        // ----------------------------------------------------------------------------
        // Update State Vector Using a Perturbation
        // ----------------------------------------------------------------------------

        void StateVector::update(const Eigen::VectorXd& perturbation) {
            slam::blockmatrix::BlockVector blk_perturb(getStateBlockSizes(), perturbation);

            // Iterate safely over the concurrent hash map
            for (auto it = states_.begin(); it != states_.end(); ++it) {
                tbb::concurrent_hash_map<slam::eval::StateKey, StateContainer, slam::eval::StateKeyHashCompare>::accessor acc;

                // Ensure safe access to the entry
                if (!states_.find(acc, it->first)) {
                throw std::runtime_error("[StateVector::update] update failed: key not found.");
                }

                auto& entry = acc->second;
                if (entry.local_block_index < 0) {
                throw std::runtime_error("[StateVector::update] update failed due to uninitialized local_block_index.");
                }

                entry.state->update(blk_perturb.at(entry.local_block_index));
            }
        }

        // ----------------------------------------------------------------------------
        // Update State Vector Using a Perturbation
        // ----------------------------------------------------------------------------

        void StateVector::clear() {
            states_.clear();
            num_block_entries_ = 0;
        }

    }  // namespace problem
}  // namespace slam
