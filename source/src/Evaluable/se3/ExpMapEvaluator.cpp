#include "source/include/Evaluable/se3/ExpMapEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace se3 {

            // ----------------------------------------------------------------------------
            // MakeShared
            // ----------------------------------------------------------------------------

            auto ExpMapEvaluator::MakeShared(const Evaluable<InType>::ConstPtr &xi) -> Ptr {
                return std::make_shared<ExpMapEvaluator>(xi);
            }

            // ----------------------------------------------------------------------------
            // ExpMapEvaluator
            // ----------------------------------------------------------------------------

            ExpMapEvaluator::ExpMapEvaluator(const Evaluable<InType>::ConstPtr &xi) : xi_(xi) {}

            // ----------------------------------------------------------------------------
            // active
            // ----------------------------------------------------------------------------

            bool ExpMapEvaluator::active() const {
                return xi_->active();
            }

            // ----------------------------------------------------------------------------
            // getRelatedVarKeys
            // ----------------------------------------------------------------------------

            void ExpMapEvaluator::getRelatedVarKeys(KeySet &keys) const {
                xi_->getRelatedVarKeys(keys);
            }

            // ----------------------------------------------------------------------------
            // value
            // ----------------------------------------------------------------------------

            auto ExpMapEvaluator::value() const -> OutType {
                return OutType(Eigen::Ref<const Eigen::VectorXd>(xi_->evaluate()));
            }

            // ----------------------------------------------------------------------------
            // forward
            // ----------------------------------------------------------------------------

            auto ExpMapEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child = xi_->forward();
                const auto value = OutType(Eigen::Ref<const Eigen::VectorXd>(child->value()));

                auto node = Node<OutType>::MakeShared(value);
                node->addChild(child);
                return node;
            }

            // ----------------------------------------------------------------------------
            // backward
            // ----------------------------------------------------------------------------

            void ExpMapEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                    const Node<OutType>::Ptr& node,
                                    StateKeyJacobians& jacs) const {
                if (!node || node->size() < 1) {
                    throw std::runtime_error("[ExpMapEvaluator::backward] Node has insufficient children.");
                }

                auto child_base = node->getChild(0);
                if (!child_base) {
                    throw std::runtime_error("[ExpMapEvaluator::backward] Null child node encountered.");
                }

                auto child = std::dynamic_pointer_cast<Node<InType>>(child_base);
                if (!child || !child->hasValue()) {
                    throw std::runtime_error("[ExpMapEval::backward] Invalid child node.");
                }

                if (xi_->active()) {
                    Eigen::MatrixXd new_lhs = lhs * slam::liemath::se3::vec2jac(node->value().vec());
                    xi_->backward(new_lhs, child, jacs);
                }
            }

            // ----------------------------------------------------------------------------
            // vec2tran
            // ----------------------------------------------------------------------------

            ExpMapEvaluator::Ptr vec2tran(const Evaluable<ExpMapEvaluator::InType>::ConstPtr &xi) {
                return ExpMapEvaluator::MakeShared(xi);
            }

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
