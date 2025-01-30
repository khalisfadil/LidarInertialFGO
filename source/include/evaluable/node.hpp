#pragma once

#include <memory>
#include <vector>
#include <utility>  // For std::move and std::forward

namespace slam {

    // -----------------------------------------------------------------------------
    /**
     * @class NodeBase
     * @brief A base class representing a generic tree node.
     *
     * This class provides basic functionality for hierarchical structures, 
     * including child node storage, retrieval, and iteration support.
     */
    class NodeBase {

        public:
        
            // -----------------------------------------------------------------------------
            /** @brief Shared pointer type for NodeBase */
            using Ptr = std::shared_ptr<NodeBase>;

            // -----------------------------------------------------------------------------
            /** @brief Constant shared pointer type for NodeBase */
            using ConstPtr = std::shared_ptr<const NodeBase>;

            // -----------------------------------------------------------------------------
            /** @brief Virtual destructor to ensure proper cleanup in derived classes */
            virtual ~NodeBase() = default;

            // -----------------------------------------------------------------------------
            /**
             * @brief Adds a child node to the current node.
             * 
             * The order of addition is preserved, meaning children are stored sequentially.
             * 
             * @param child Shared pointer to the child node to be added.
             */
            void addChild(const Ptr& child) { children_.emplace_back(child); }

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves the number of child nodes.
             * 
             * @return The number of child nodes stored in this node.
             */
            size_t size() const { return children_.size(); }

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves a child node at a specified index safely.
             * 
             * This method returns a shared pointer to the child if the index is valid, 
             * otherwise, it returns `nullptr`, preventing out-of-bounds errors.
             * 
             * @param index The position of the child node.
             * @return Shared pointer to the child node, or `nullptr` if the index is invalid.
             */
            Ptr getChild(size_t index) const {
                return (index < children_.size()) ? children_[index] : nullptr;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Provides range-based iteration support.
             * 
             * Enables traversal of child nodes using modern C++ range-based loops.
             * Example:
             * @code
             * for (const auto& child : *node) {
             *   // Process child node
             * }
             * @endcode
             */
            auto begin() { return children_.begin(); }
            auto end() { return children_.end(); }
            auto begin() const { return children_.begin(); }
            auto end() const { return children_.end(); }

        private:

            // -----------------------------------------------------------------------------
            std::vector<Ptr> children_;  ///< Stores child nodes in a sequential manner.

    };

    // -----------------------------------------------------------------------------
    /**
     * @class Node<T>
     * @brief A templated node class that extends NodeBase to store typed values.
     * 
     * This class allows tree nodes to store a specific data type `T`, making it useful 
     * for various applications such as factor graphs, hierarchical data storage, and more.
     * 
     * @tparam T The type of data stored in the node.
     */
    template <class T>
    class Node : public NodeBase {

        public:

            // -----------------------------------------------------------------------------
            /** @brief Shared pointer type for Node<T> */
            using Ptr = std::shared_ptr<Node<T>>;

            // -----------------------------------------------------------------------------
            /** @brief Constant shared pointer type for Node<T> */
            using ConstPtr = std::shared_ptr<const Node<T>>;

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory method to create a shared instance of Node<T>.
             * 
             * This method supports both l-value and r-value references, enabling efficient 
             * object creation without unnecessary copies.
             * 
             * @tparam U Type of the value being stored (supports derived types).
             * @param value The value to be stored in the node.
             * @return Shared pointer to the created node instance.
             */
            template <typename U>
            static Ptr MakeShared(U&& value) {
                return std::make_shared<Node<T>>(std::forward<U>(value));
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs a Node<T> with a given value.
             * 
             * Supports both copy and move semantics to handle different types efficiently.
             * 
             * @tparam U Type of the value being stored.
             * @param value The value to be stored (can be moved or copied).
             */
            template <typename U>
            explicit Node(U&& value) : value_(std::forward<U>(value)) {}

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves the stored value.
             * 
             * @return A constant reference to the stored value.
             */
            const T& value() const { return value_; }

        private:

            // -----------------------------------------------------------------------------
            T value_;  ///< The value stored within the node.

    };

}  // namespace slam
