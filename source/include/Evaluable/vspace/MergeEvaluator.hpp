#pragma once

#include <memory>
#include <Eigen/Core>

#include "Evaluable/Evaluable.hpp"
#include "Evaluable/Node.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // -----------------------------------------------------------------------------
            /**
             * @class MergeEvaluator
             * @brief Concatenates two vector-valued functions into a single output vector.
             *
             * **Functionality:**
             * - Computes **[v1; v2]**, where `v1` and `v2` are stacked vertically.
             * - Supports **forward propagation** (computing values).
             * - Supports **backward propagation** (Jacobian computation).
             * - Tracks dependencies for **automatic differentiation**.
             *
             * @tparam DIM1 Dimensionality of the first input vector.
             * @tparam DIM2 Dimensionality of the second input vector.
             */
            template <int DIM1, int DIM2>
            class MergeEvaluator : public Evaluable<Eigen::Matrix<double, DIM1 + DIM2, 1>> {
            public:
                using Ptr = std::shared_ptr<MergeEvaluator>;
                using ConstPtr = std::shared_ptr<const MergeEvaluator>;

                using In1Type = Eigen::Matrix<double, DIM1, 1>;
                using In2Type = Eigen::Matrix<double, DIM2, 1>;
                using OutType = Eigen::Matrix<double, DIM1 + DIM2, 1>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method for creating a shared instance.
                 * @param v1 First function to be merged.
                 * @param v2 Second function to be merged.
                 */
                static Ptr MakeShared(const typename Evaluable<In1Type>::ConstPtr& v1,
                                      const typename Evaluable<In2Type>::ConstPtr& v2) {
                    return std::make_shared<MergeEvaluator>(v1, v2);
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for MergeEvaluator.
                 * @param v1 First function to be merged.
                 * @param v2 Second function to be merged.
                 */
                MergeEvaluator(const typename Evaluable<In1Type>::ConstPtr& v1,
                               const typename Evaluable<In2Type>::ConstPtr& v2)
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
                /** @brief Computes the merged output value. */
                OutType value() const override {
                    OutType value = OutType::Zero();
                    value.topRows(DIM1) = v1_->value();
                    value.bottomRows(DIM2) = v2_->value();
                    return value;
                }

                // -----------------------------------------------------------------------------
                /** @brief Creates a computation node for forward propagation. */
                typename Node<OutType>::Ptr forward() const override {
                    const auto child1 = v1_->forward();
                    const auto child2 = v2_->forward();

                    OutType value = OutType::Zero();
                    value.topRows(DIM1) = child1->value();
                    value.bottomRows(DIM2) = child2->value();

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
                    if (v1_->active()) {
                        const auto child1 = std::static_pointer_cast<Node<In1Type>>(node->getChild(0));
                        v1_->backward(lhs.leftCols(DIM1), child1, jacs);
                    }

                    if (v2_->active()) {
                        const auto child2 = std::static_pointer_cast<Node<In2Type>>(node->getChild(1));
                        v2_->backward(lhs.rightCols(DIM2), child2, jacs);
                    }
                }

            private:
                const typename Evaluable<In1Type>::ConstPtr v1_;  ///< First input function.
                const typename Evaluable<In2Type>::ConstPtr v2_;  ///< Second input function.
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Convenience function for creating a MergeEvaluator.
             * @param v1 First function to be merged.
             * @param v2 Second function to be merged.
             */
            template <int DIM1, int DIM2>
            typename MergeEvaluator<DIM1, DIM2>::Ptr merge(
                const typename Evaluable<typename MergeEvaluator<DIM1, DIM2>::In1Type>::ConstPtr& v1,
                const typename Evaluable<typename MergeEvaluator<DIM1, DIM2>::In2Type>::ConstPtr& v2) {
                return MergeEvaluator<DIM1, DIM2>::MakeShared(v1, v2);
            }

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam