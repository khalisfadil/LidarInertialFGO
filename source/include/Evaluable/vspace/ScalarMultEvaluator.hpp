#pragma once

#include <memory>
#include <Eigen/Core>

#include "Evaluable/Evaluable.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // -----------------------------------------------------------------------------
            /**
             * @class ScalarMultEvaluator
             * @brief Evaluates the element-wise scalar multiplication of a vector function.
             *
             * **Functionality:**
             * - Computes **s * v**, where `s` is a scalar and `v` is a vector-valued function.
             * - Supports forward and backward propagation for **automatic differentiation**.
             *
             * @tparam DIM Dimensionality of the input vector (can be dynamic).
             */
            template <int DIM = Eigen::Dynamic>
            class ScalarMultEvaluator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
            public:
                using Ptr = std::shared_ptr<ScalarMultEvaluator>;
                using ConstPtr = std::shared_ptr<const ScalarMultEvaluator>;

                using InType = Eigen::Matrix<double, DIM, 1>;
                using OutType = Eigen::Matrix<double, DIM, 1>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method for creating a shared instance.
                 * @param v Input vector function.
                 * @param s Scalar multiplier.
                 */
                static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v, const double& s) {
                    return std::make_shared<ScalarMultEvaluator>(v, s);
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for ScalarMultEvaluator.
                 * @param v Input vector function.
                 * @param s Scalar multiplier.
                 */
                ScalarMultEvaluator(const typename Evaluable<InType>::ConstPtr& v, const double& s)
                    : v_(v), s_(s) {}

                // -----------------------------------------------------------------------------
                /** @brief Checks if this evaluator depends on active state variables. */
                bool active() const override {
                    return v_->active();
                }

                // -----------------------------------------------------------------------------
                /** @brief Retrieves keys of related state variables. */
                void getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const override {
                    v_->getRelatedVarKeys(keys);
                }

                // -----------------------------------------------------------------------------
                /** @brief Computes the scalar-multiplied output value. */
                OutType value() const override {
                    return s_ * v_->value();
                }

                // -----------------------------------------------------------------------------
                /** @brief Creates a computation node for forward propagation. */
                typename Node<OutType>::Ptr forward() const override {
                    const auto child = v_->forward();
                    const auto value = s_ * child->value();
                    const auto node = Node<OutType>::MakeShared(value);
                    node->addChild(child);
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
                    const auto child = std::static_pointer_cast<Node<InType>>(node->getChild(0));
                    if (v_->active()) {
                        v_->backward(s_ * lhs, child, jacs);
                    }
                }

            private:
                const typename Evaluable<InType>::ConstPtr v_;  ///< Input vector function.
                const double s_;  ///< Scalar multiplier.
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Convenience function for creating a ScalarMultEvaluator instance.
             * @param v Input vector function.
             * @param s Scalar multiplier.
             * @return Shared pointer to the created ScalarMultEvaluator.
             */
            template <int DIM>
            typename ScalarMultEvaluator<DIM>::Ptr smult(
                const typename Evaluable<typename ScalarMultEvaluator<DIM>::InType>::ConstPtr& v,
                const double& s) {
                return ScalarMultEvaluator<DIM>::MakeShared(v, s);
            }

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam