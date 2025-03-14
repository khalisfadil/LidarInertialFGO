#pragma once

#include <memory>
#include <Eigen/Core>

#include "Core/Evaluable/Evaluable.hpp"
#include "Core/Evaluable/Node.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // -----------------------------------------------------------------------------
            /**
             * @class AdditionEvaluator
             * @brief Computes element-wise addition of two vector-valued functions.
             *
             * **Functionality:**
             * - Computes **v1 + v2** at runtime.
             * - Supports **forward propagation** (computing values).
             * - Supports **backward propagation** (Jacobian computation).
             * - Tracks dependencies for **automatic differentiation**.
             *
             * @tparam DIM Dimensionality of input vectors (can be dynamic).
             */
            template <int DIM = Eigen::Dynamic>
            class AdditionEvaluator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
            public:
                using Ptr = std::shared_ptr<AdditionEvaluator>;
                using ConstPtr = std::shared_ptr<const AdditionEvaluator>;

                using InType = Eigen::Matrix<double, DIM, 1>;
                using OutType = Eigen::Matrix<double, DIM, 1>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method for creating a shared instance.
                 * @param v1 First function to be added.
                 * @param v2 Second function to be added.
                 */
                static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v1,
                                      const typename Evaluable<InType>::ConstPtr& v2) {
                    return std::make_shared<AdditionEvaluator>(v1, v2);
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for AdditionEvaluator.
                 * @param v1 First function to be added.
                 * @param v2 Second function to be added.
                 */
                AdditionEvaluator(const typename Evaluable<InType>::ConstPtr& v1,
                                  const typename Evaluable<InType>::ConstPtr& v2)
                    : v1_(v1), v2_(v2) {}

                // -----------------------------------------------------------------------------
                /** @brief Determines if this evaluator depends on active variables. */
                bool active() const override {
                    return v1_->active() || v2_->active();
                }

                // -----------------------------------------------------------------------------
                /** @brief Retrieves keys of related state variables. */
                void getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const override {
                    v1_->getRelatedVarKeys(keys);
                    v2_->getRelatedVarKeys(keys);
                }

                // -----------------------------------------------------------------------------
                /** @brief Computes the summed output value. */
                OutType value() const override {
                    return v1_->value() + v2_->value();
                }

                // -----------------------------------------------------------------------------
                /** @brief Creates a computation node for forward propagation. */
                typename Node<OutType>::Ptr forward() const override {
                    const auto child1 = v1_->forward();
                    const auto child2 = v2_->forward();
                    const auto value = child1->value() + child2->value();
                    const auto node = Node<OutType>::MakeShared(value);
                    node->addChild(child1);
                    node->addChild(child2);
                    return node;
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Performs backpropagation to accumulate Jacobians.
                 * @param lhs Left-hand-side weight matrix.
                 * @param node Computation node from `forward()`.
                 * @param jacs Storage for accumulated Jacobians.
                 */
                void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                              const typename Node<OutType>::Ptr& node,
                              StateKeyJacobians& jacs) const override {
                    const auto child1 = std::static_pointer_cast<Node<InType>>(node->getChild(0));
                    const auto child2 = std::static_pointer_cast<Node<InType>>(node->getChild(1));

                    if (v1_->active()) {
                        v1_->backward(lhs, child1, jacs);
                    }

                    if (v2_->active()) {
                        v2_->backward(lhs, child2, jacs);
                    }
                }

            private:
                const typename Evaluable<InType>::ConstPtr v1_;  ///< First input function.
                const typename Evaluable<InType>::ConstPtr v2_;  ///< Second input function.
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Convenience function for creating an AdditionEvaluator.
             * @param v1 First function to be added.
             * @param v2 Second function to be added.
             */
            template <int DIM>
            typename AdditionEvaluator<DIM>::Ptr add(
                const typename Evaluable<typename AdditionEvaluator<DIM>::InType>::ConstPtr& v1,
                const typename Evaluable<typename AdditionEvaluator<DIM>::InType>::ConstPtr& v2) {
                return AdditionEvaluator<DIM>::MakeShared(v1, v2);
            }

        } // namespace vspace
    }  // namespace eval
}  // namespace slam