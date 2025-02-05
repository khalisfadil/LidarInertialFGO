#pragma once

#include <Eigen/Dense>
#include <include/evaluable/evaluable.hpp>
#include <include/evaluable/stateKey.hpp>

namespace slam {

    // -----------------------------------------------------------------------------
    /**
     * @brief Base class for all state variables in the optimization framework.
     * 
     * This class provides a generic interface for state variables, including
     * key-based identification, locking mechanisms, and perturbation dimensions.
     */
    class StateVariableBase {

        public:

            using Ptr = std::shared_ptr<StateVariableBase>;
            using ConstPtr = std::shared_ptr<const StateVariableBase>;

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructor for the base state variable.
             * @param perturb_dim The dimension of the perturbation applied to the state.
             * @param name Optional name for the state variable.
             */
            StateVariableBase(const unsigned int perturb_dim, const std::string& name = "")
                : perturb_dim_(perturb_dim), name_(name), key_(NewStateKey()) {}

            virtual ~StateVariableBase() = default;

            // -----------------------------------------------------------------------------
            /**
             * @brief Returns the name of the state variable.
             */
            std::string name() const { return name_; }

            // -----------------------------------------------------------------------------
            /**
             * @brief Updates this state variable from a perturbation.
             * @param perturbation The perturbation vector to apply.
             * @return True if the update is successful, false if the state is locked.
             */
            virtual bool update(const Eigen::VectorXd& perturbation) = 0;

            // -----------------------------------------------------------------------------
            /**
             * @brief Returns a clone of the state variable.
             * @return A shared pointer to the cloned state.
             */
            virtual Ptr clone() const = 0;

            // -----------------------------------------------------------------------------
            /**
             * @brief Copies the state value from another instance.
             * @param other The other state variable to copy from.
             */
            virtual void setFromCopy(const ConstPtr& other) = 0;

            // -----------------------------------------------------------------------------
            /**
             * @brief Returns the unique key associated with this state variable.
             */
            StateKey key() const { return key_; }

            // -----------------------------------------------------------------------------
            /**
             * @brief Returns the perturbation dimension of this state variable.
             */
            unsigned int perturb_dim() const { return perturb_dim_; }

            // -----------------------------------------------------------------------------
            /**
             * @brief Checks if the state variable is locked (i.e., not updated in optimization).
             */
            bool locked() const { return locked_; }

            // -----------------------------------------------------------------------------
            /**
             * @brief Allows modification of the lock status.
             */
            void setLocked(bool lock_status) { locked_ = lock_status; }

        private:

            const unsigned int perturb_dim_;  ///< Dimension of the perturbation
            const std::string name_;          ///< Name of the state variable
            const StateKey key_; ///< Unique identifier for the state
            bool locked_ = false; ///< Indicates whether the state is locked (non-optimizable)
    };

}  // namespace slam
