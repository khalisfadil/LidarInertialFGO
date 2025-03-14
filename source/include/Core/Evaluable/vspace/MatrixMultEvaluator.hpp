#pragma once

#include <Eigen/Core>
#include "Core/Evaluable/Evaluable.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // -----------------------------------------------------------------------------
            /**
             * @class MatrixMultEvaluator
             * @brief Evaluates the product of a matrix and a vector function.
             *
             * **Functionality:**
             * - Computes **s * v** where `s` is a matrix and `v` is an evaluable function.
             * - Supports **forward propagation** (computing values).
             * - Supports **backward propagation** (Jacobian computation).
             * - Optimized for **memory efficiency** and **lazy evaluation**.
             *
             * @tparam ROW Number of rows in the output vector (can be dynamic).
             * @tparam COL Number of columns in the input vector (default = ROW).
             */
            template <int ROW = Eigen::Dynamic, int COL = ROW>
            class MatrixMultEvaluator : public Evaluable<Eigen::Matrix<double, ROW, 1>> {
            public:
                using Ptr = std::shared_ptr<MatrixMultEvaluator>;
                using ConstPtr = std::shared_ptr<const MatrixMultEvaluator>;

                using MatType = Eigen::Matrix<double, ROW, COL>;
                using InType = Eigen::Matrix<double, COL, 1>;
                using OutType = Eigen::Matrix<double, ROW, 1>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method for creating a shared instance.
                 * @param v The input function to be multiplied.
                 * @param s The matrix used for multiplication.
                 */
                static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v,
                                      const Eigen::Ref<const MatType>& s) {
                    return std::make_shared<MatrixMultEvaluator>(v, s);
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for MatrixMultEvaluator.
                 * @param v The input function to be multiplied.
                 * @param s The matrix used for multiplication.
                 */
                MatrixMultEvaluator(const typename Evaluable<InType>::ConstPtr& v,
                                    const Eigen::Ref<const MatType>& s)
                    : v_(v), s_(s) {}

                // -----------------------------------------------------------------------------
                /** @brief Determines if this evaluator depends on active variables. */
                bool active() const override {
                    return v_->active();
                }

                // -----------------------------------------------------------------------------
                /** @brief Retrieves keys of related state variables. */
                void getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const override {
                    v_->getRelatedVarKeys(keys);
                }

                // -----------------------------------------------------------------------------
                /** @brief Computes the matrix-vector product output value. */
                OutType value() const override {
                    OutType result;
                    result.noalias() = s_ * v_->value();
                    return result;
                }

                // -----------------------------------------------------------------------------
                /** @brief Creates a computation node for forward propagation. */
                typename Node<OutType>::Ptr forward() const override {
                    const auto child = v_->forward();
                    OutType value;
                    value.noalias() = s_ * child->value();
                    const auto node = Node<OutType>::MakeShared(Eigen::Ref<const OutType>(value));
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
                        v_->backward(lhs * s_.eval(), child, jacs);
                    }
                }

            private:
                const typename Evaluable<InType>::ConstPtr v_;  ///< Input function.
                const Eigen::Ref<const MatType> s_;  ///< Matrix used for multiplication (reference to avoid copies).
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Convenience function for creating a MatrixMultEvaluator.
             * @param v The input function to be multiplied.
             * @param s The matrix used for multiplication.
             */
            template <int ROW, int COL = ROW>
            typename MatrixMultEvaluator<ROW, COL>::Ptr mmult(
                const typename Evaluable<typename MatrixMultEvaluator<ROW, COL>::InType>::ConstPtr& v,
                const typename MatrixMultEvaluator<ROW, COL>::MatType& s) {
                return MatrixMultEvaluator<ROW, COL>::MakeShared(v, s);
            }

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam