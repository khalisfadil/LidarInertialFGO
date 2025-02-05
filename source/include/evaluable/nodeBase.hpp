#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <stdexcept>

namespace slam {

    // -----------------------------------------------------------------------------
    /**
     * @class NodeBase
     * @brief A base class representing a generic tree node with thread-safe access.
     *
     * @note This class requires that instances are always managed by std::shared_ptr
     *       to avoid exceptions from shared_from_this().
     */
    class NodeBase : public std::enable_shared_from_this<NodeBase> {

        public:
            using Ptr = std::shared_ptr<NodeBase>;
            using ConstPtr = std::shared_ptr<const NodeBase>;

            virtual ~NodeBase() = default;

            // -----------------------------------------------------------------------------
            /**
             * @brief Adds a child node to the current node (thread-safe).
             * @throws std::runtime_error if this NodeBase is not owned by a shared_ptr (bad_weak_ptr).
             */
            void addChild(const Ptr& child) {
                if (!child) {
                    return;
                }
                // Best practice: ensure that this object is managed by a shared_ptr
                // so that shared_from_this() doesn't throw bad_weak_ptr.
                std::shared_ptr<NodeBase> parent_shared;
                try {
                    parent_shared = shared_from_this();
                } catch (const std::bad_weak_ptr&) {
                    throw std::runtime_error("[NodeBase::addChild] Object not owned by a std::shared_ptr");
                }

                // Set child's parent pointer
                child->parent_ = parent_shared;

                // Lock for writing
                {
                    std::unique_lock<std::shared_mutex> lock(children_mutex_);
                    children_.push_back(child);
                    children_size_.fetch_add(1, std::memory_order_relaxed);
                }
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves the number of child nodes in a lock-free manner.
             */
            size_t size() const {
                return children_size_.load(std::memory_order_acquire);
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves a child node at index, or nullptr if out-of-bounds.
             */
            Ptr getChild(size_t index) const {
                std::shared_lock<std::shared_mutex> lock(children_mutex_);
                if (index >= children_.size()) {
                    return nullptr;
                }
                return children_[index];
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Retrieves the parent node (may be nullptr if none).
             */
            std::shared_ptr<NodeBase> getParent() const {
                return parent_.lock();
            }

        protected:
        
            // Recommend making constructor protected so users cannot instantiate on the stack
            NodeBase() = default;

            // Data
            std::vector<Ptr> children_;
            mutable std::shared_mutex children_mutex_;
            std::weak_ptr<NodeBase> parent_;
            std::atomic<size_t> children_size_{0};
    };

}  // namespace slam
