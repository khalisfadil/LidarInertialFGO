#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <typeinfo>

#include "Evaluable/NodeBase.hpp"

namespace slam {
    namespace eval{

        // -----------------------------------------------------------------------------
        /**
         * @class Node<T>
         * @brief A templated node class that extends NodeBase to store typed values.
         */
        template <class T>
        class Node : public NodeBase {

            public:

                using Ptr = std::shared_ptr<Node<T>>;
                using ConstPtr = std::shared_ptr<const Node<T>>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create a shared instance of Node<T>.
                 *        Forwards any args to Node's constructor.
                 */
                template <typename... Args>
                static Ptr MakeShared(Args&&... args) {
                    return std::make_shared<Node<T>>(std::forward<Args>(args)...);
                }

                Node() = default;

                template <typename U>
                explicit Node(U&& value)
                    : NodeBase(), value_(std::forward<U>(value))
                {}

                bool hasValue() const noexcept {
                    return value_.has_value();
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves the stored value (read-only).
                 * @throws std::logic_error if the value is not set.
                 */
                const T& value() const {
                    if (!value_) {
                        throw std::logic_error("[Node<T>::value] Uninitialized value of type "
                                            + std::string(typeid(T).name()));
                    }
                    return *value_;
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves the stored value (modifiable).
                 * @throws std::logic_error if the value is not set.
                 */
                T& mutableValue() {
                    if (!value_) {
                        throw std::logic_error("[Node<T>::mutableValue] Uninitialized value of type "
                                            + std::string(typeid(T).name()));
                    }
                    return *value_;
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Sets a new value, replacing the existing one.
                 *        Removed the == check to avoid compile issues for non-equality-comparable T.
                 */
                template <typename U>
                void setValue(U&& newValue) {
                    value_.emplace(std::forward<U>(newValue));
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Resets the stored value, making it uninitialized.
                 * @return true if a reset occurred; false if already empty.
                 */
                bool resetValue() noexcept {
                    if (!value_) {
                        return false;
                    }
                    value_.reset();
                    return true;
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Returns a pointer to the stored value or nullptr if uninitialized.
                 */
                const T* valuePtr() const noexcept {
                    return value_ ? &(*value_) : nullptr;
                }

            private:
            
                std::optional<T> value_;
        };
    } // namespace eval
}  // namespace slam
