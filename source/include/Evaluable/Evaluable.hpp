#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <Eigen/Core>
#include <unordered_set>
#include <optional>
#include <stdexcept>
#include <typeinfo>

#include <include/Evaluable/StateKeyJacobians.hpp>
#include <include/Evaluable/Node.hpp>

namespace slam {
    namespace eval{

        //-----------------------------------------------------------------------------
        /**
         * @class Evaluable
         * @brief Abstract base for an evaluable function with automatic differentiation support.
         *
         * @tparam T The output type (e.g. a scalar or Eigen::VectorXd).
         */
        template <class T>
        class Evaluable {

            public:

                using Ptr = std::shared_ptr<Evaluable<T>>;
                using ConstPtr = std::shared_ptr<const Evaluable<T>>;
                using KeySet = std::unordered_set<StateKey, StateKeyHash>;

                virtual ~Evaluable() = default;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Evaluates the function, possibly using a cached value unless force=true.
                 * @throws std::runtime_error if `value()` throws internally.
                 */
                [[nodiscard]] T evaluate(bool force = false) const {
                    // NOTE: This method is NOT thread-safe if called concurrently on the same object.
                    if (force || !cached_value_) {
                        try {
                            cached_value_ = this->value();
                        } catch (const std::exception& e) {
                            throw std::runtime_error(std::string("[Evaluable<T>::evaluate] Error in value() of type ")
                                + typeid(T).name() + ": " + e.what());
                        }
                    }
                    return *cached_value_;
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Clears the cached value, forcing re-evaluation next time.
                 */
                void resetCache() const noexcept {
                    cached_value_.reset();
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Evaluates the function and accumulates its Jacobians.
                 * @param lhs  Left-hand-side weight matrix.
                 * @param jacs Container for storing computed Jacobians.
                 * @return The computed value.
                 * @throws std::logic_error if forward() returns nullptr or node has no value.
                 * @throws std::runtime_error if backward() throws internally.
                 */
                [[nodiscard]] T evaluate(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                        StateKeyJacobians& jacs) const {
                    auto end_node = this->forward();
                    if (!end_node) {
                        throw std::logic_error("[Evaluable<T>::evaluate] forward() returned nullptr for type "
                                            + std::string(typeid(T).name()));
                    }

                    // Safety check: does the returned node actually contain a value?
                    if (!end_node->hasValue()) {
                        throw std::logic_error("[Evaluable<T>::evaluate] Node returned by forward() has no value for type "
                                            + std::string(typeid(T).name()));
                    }

                    // Perform backward accumulation
                    try {
                        this->backward(lhs, end_node, jacs);
                    } catch (const std::exception& e) {
                        throw std::runtime_error(std::string("[Evaluable<T>::evaluate] Error in backward() for type ")
                                                + typeid(T).name() + ": " + e.what());
                    }

                    return end_node->value();
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Checks if the function depends on active state variables.
                 */
                virtual bool active() const = 0;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Gathers the keys of state variables that affect this function.
                 */
                virtual void getRelatedVarKeys(KeySet& keys) const = 0;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes the value of this function (pure virtual).
                 */
                virtual T value() const = 0;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Forward pass: builds a Node<T> to hold the computed value for use in backward differentiation.
                 * @return Node<T>::Ptr containing the computed value (must not be nullptr).
                 */
                virtual typename Node<T>::Ptr forward() const = 0;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Backward pass: accumulates Jacobians in jacs using the given lhs and the Node returned by forward().
                 * @param lhs   Left-hand-side weight matrix.
                 * @param node  The Node containing the forward() result.
                 * @param jacs  Accumulation container for Jacobians.
                 */
                virtual void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                    const typename Node<T>::Ptr& node,
                                    StateKeyJacobians& jacs) const = 0;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Prints all related state keys for debugging.
                 * @param os Output stream (default std::cout).
                 * @param prefix An optional prefix to show in the log.
                 */
                void printKeys(std::ostream& os = std::cout,
                            const std::string& prefix = "[Evaluable]") const {
                    KeySet keys;
                    getRelatedVarKeys(keys);
                    os << prefix << " { \"Related StateKeys\": [ ";
                    bool first = true;
                    for (auto& key : keys) {
                        if (!first) os << ", ";
                        os << key;
                        first = false;
                    }
                    os << " ] }\n";
                }

            private:

                // Cache of the last computed value (not thread-safe if used concurrently!)
                mutable std::optional<T> cached_value_;
        };
    } // namespace eval
}  // namespace slam
