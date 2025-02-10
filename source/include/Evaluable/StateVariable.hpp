#pragma once

#include <memory>  // Ensure this is at the top
#include <Eigen/Dense>
#include "source/include/Evaluable/StateVariableBase.hpp"
#include "source/include/Evaluable/StateKey.hpp"

namespace slam {
    namespace eval{
        // -----------------------------------------------------------------------------
        /**
         * @brief Templated class representing a state variable of type T.
         * 
         * This class stores a value and integrates with the factor graph framework
         * by implementing the Evaluable<T> interface.
         */
        template <class T>
        class StateVariable : public StateVariableBase, public Evaluable<T> {

            public:

                using Ptr = std::shared_ptr<StateVariable<T>>;
                using ConstPtr = std::shared_ptr<const StateVariable<T>>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs a state variable with an initial value.
                 * @param value The initial value of the state variable.
                 * @param perturb_dim The dimension of perturbation applied to the state.
                 * @param name Optional name for the state variable.
                 */
                StateVariable(const T& value, const unsigned int perturb_dim,
                        const std::string& name = "")
                    : StateVariableBase(perturb_dim, name), value_(value) {}

                // -----------------------------------------------------------------------------
                /**
                 * @brief Copies the value from another state variable instance.
                 * @param other The other state variable to copy from.
                 * @throws std::runtime_error if keys do not match.
                 */
                void setFromCopy(const StateVariableBase::ConstPtr& other) override {
                    if (key() != other->key())
                        throw std::runtime_error("StateVariable::setFromCopy: keys do not match");
                    value_ = std::static_pointer_cast<const StateVariable<T>>(other)->value_;
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Checks if the state variable is active in the optimization.
                 * @return True if the variable is not locked, otherwise false.
                 */
                bool active() const override { return !locked(); }

                using KeySet = typename Evaluable<T>::KeySet;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves the related variable keys for optimization.
                 * @param keys The set to store related variable keys.
                 */
                void getRelatedVarKeys(KeySet& keys) const override {
                    if (!locked()) keys.insert(key());
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Returns the current value of the state variable.
                 */
                T value() const override { return value_; }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Performs forward computation for factor graph evaluation.
                 * @return A shared pointer to the computed node.
                 */
                typename Node<T>::Ptr forward() const override {
                    return Node<T>::MakeShared(value_);
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes the Jacobian for the optimization framework.
                 */
                void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs, const typename Node<T>::Ptr& node,
                                StateKeyJacobians& jacs) const override {
                    if (active()) jacs.add(key(), lhs);
                }

            protected:

                T value_; ///< The actual value of the state variable.

        };
    } // namespace eval
}  // namespace slam
