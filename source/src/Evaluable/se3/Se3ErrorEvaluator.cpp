#include "source/include/Evaluable/se3/Se3ErrorEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace se3 {

            // ----------------------------------------------------------------------------
            // MakeShared
            // ----------------------------------------------------------------------------

            auto SE3ErrorEvaluator::MakeShared(const Evaluable<InType>::ConstPtr& T_ab,
                                        const InType& T_ab_meas) -> Ptr {
                return std::make_shared<SE3ErrorEvaluator>(T_ab, T_ab_meas);
            }

            // ----------------------------------------------------------------------------
            // SE3ErrorEvaluator
            // ----------------------------------------------------------------------------

            SE3ErrorEvaluator::SE3ErrorEvaluator(const Evaluable<InType>::ConstPtr& T_ab,
                                    const InType& T_ab_meas)
                : T_ab_(T_ab), T_ab_meas_(T_ab_meas) {}

            // ----------------------------------------------------------------------------
            // active
            // ----------------------------------------------------------------------------

            bool SE3ErrorEvaluator::active() const {
                return T_ab_->active();
            }

            // ----------------------------------------------------------------------------
            // getRelatedVarKeys
            // ----------------------------------------------------------------------------

            void SE3ErrorEvaluator::getRelatedVarKeys(KeySet& keys) const {
                T_ab_->getRelatedVarKeys(keys);
            }

            // ----------------------------------------------------------------------------
            // value
            // ----------------------------------------------------------------------------

            auto SE3ErrorEvaluator::value() const -> OutType {
                return (T_ab_meas_ * T_ab_->evaluate().inverse()).vec();
            }

            // ----------------------------------------------------------------------------
            // forward
            // ----------------------------------------------------------------------------

            auto SE3ErrorEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child = T_ab_->forward();
                const auto value = (T_ab_meas_ * child->value().inverse()).vec();

                auto node = Node<OutType>::MakeShared(value);
                node->addChild(child);
                return node;
            }

            // ----------------------------------------------------------------------------
            // backward
            // ----------------------------------------------------------------------------

            void SE3ErrorEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                        const Node<OutType>::Ptr& node,
                                        StateKeyJacobians& jacs) const {
                if (!node || node->size() < 1) {
                    throw std::runtime_error("[SE3ErrorEvaluator::backward] Node has insufficient children.");
                }

                auto child_base = node->getChild(0);
                if (!child_base) {
                    throw std::runtime_error("[SE3ErrorEvaluator::backward] Null child node encountered.");
                }

                auto child = std::dynamic_pointer_cast<Node<InType>>(child_base);
                if (!child || !child->hasValue()) {
                    throw std::runtime_error("[SE3ErrorEvaluator::backward] Invalid child node.");
                }

                if (T_ab_->active()) {
                    Eigen::MatrixXd new_lhs = lhs * (-1.0) * slam::liemath::se3::vec2jac(node->value());
                    T_ab_->backward(new_lhs, child, jacs);
                }
            }

            // ----------------------------------------------------------------------------
            // se3_error
            // ----------------------------------------------------------------------------

            SE3ErrorEvaluator::Ptr se3_error(const Evaluable<SE3ErrorEvaluator::InType>::ConstPtr& T_ab,
                                        const SE3ErrorEvaluator::InType& T_ab_meas) {
                return SE3ErrorEvaluator::MakeShared(T_ab, T_ab_meas);
            }

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
