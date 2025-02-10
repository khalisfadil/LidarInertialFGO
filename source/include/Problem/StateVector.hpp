#pragma once

#include <vector>
#include <memory>
#include <tbb/concurrent_hash_map.h>
#include <Eigen/Dense>

#include "source/include/Evaluable/StateKeyJacobians.hpp"
#include "source/include/Evaluable/StateKey.hpp"
#include "source/include/Evaluable/StateVariableBase.hpp"

namespace slam {
  namespace problem {

    // -----------------------------------------------------------------------------
    /**
       * @class StateVector
       * @brief Container for managing state variables in optimization.
       * 
       * Provides:
       * - **Thread-safe storage** using `tbb::concurrent_hash_map`.
       * - **Ordered block indexing** for optimization.
       * - **Efficient state updates** with perturbations.
       */
    class StateVector {
      public:

        // -----------------------------------------------------------------------------
        /** @brief Convenience typedefs */
        using Ptr = std::shared_ptr<StateVector>;
        using ConstPtr = std::shared_ptr<const StateVector>;

        // -----------------------------------------------------------------------------
        /** @brief Default constructor */
        StateVector() = default;

        // -----------------------------------------------------------------------------
        /** @brief Explicit copy constructor */
        StateVector(const StateVector& other);

        // -----------------------------------------------------------------------------
        /** @brief Factory method to create a shared instance */
        static Ptr MakeShared() {
          return std::make_shared<StateVector>();
        }

        // -----------------------------------------------------------------------------
        /** @brief Performs a deep copy of the state vector */
        [[nodiscard]] StateVector clone() const;

        // -----------------------------------------------------------------------------
        /**
         * @brief Copies values from another state vector.
         * @param other The state vector to copy from.
         * @throws std::runtime_error if structures do not match.
         */
        void copyValues(const StateVector& other);

        // -----------------------------------------------------------------------------
        /** @brief Adds a state variable to the vector. */
        void addStateVariable(const slam::eval::StateVariableBase::Ptr& statevar);

        // -----------------------------------------------------------------------------
        /** @brief Checks if a state variable exists. */
        [[nodiscard]] bool hasStateVariable(const slam::eval::StateKey& key) const noexcept;

        // -----------------------------------------------------------------------------
        /** @brief Retrieves a state variable by key. */
        [[nodiscard]] slam::eval::StateVariableBase::ConstPtr getStateVariable(const slam::eval::StateKey& key) const;

        // -----------------------------------------------------------------------------
        /** @brief Returns the total number of state variables. */
        [[nodiscard]] unsigned int getNumberOfStates() const noexcept;

        // -----------------------------------------------------------------------------
        /** @brief Returns the block index of a specific state. */
        [[nodiscard]] int getStateBlockIndex(const slam::eval::StateKey& key) const;

        // -----------------------------------------------------------------------------
        /** @brief Returns an ordered list of block sizes. */
        [[nodiscard]] std::vector<unsigned int> getStateBlockSizes() const;

        // -----------------------------------------------------------------------------
        /** @brief Returns the total size of the state vector. */
        [[nodiscard]] unsigned int getStateSize() const noexcept;
        
        // -----------------------------------------------------------------------------
        /** @brief Updates the state vector using a perturbation. */
        void update(const Eigen::VectorXd& perturbation);

    private:
        // -----------------------------------------------------------------------------
        /** @brief Container for state variables and indexing. */
        struct StateContainer {
          slam::eval::StateVariableBase::Ptr state;  ///< The actual state variable.
          int local_block_index = -1;    ///< Block index in the active state (-1 if inactive).
        };

        // -----------------------------------------------------------------------------
        /** @brief Thread-safe storage for state variables. */
        using StateMap = tbb::concurrent_hash_map<slam::eval::StateKey, StateContainer, slam::eval::StateKeyHash>;
        StateMap states_;

        // -----------------------------------------------------------------------------
        /** @brief Total number of block entries in the state vector (atomic for thread safety). */
        std::atomic<unsigned int> num_block_entries_{0};
    };

  }  // namespace problem
}  // namespace slam
 