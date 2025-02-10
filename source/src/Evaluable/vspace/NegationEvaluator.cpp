#include "source/include/Evaluable/vspace/NegationEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // ----------------------------------------------------------------------------
            // MakeShared
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto NegationEvaluator<DIM>::MakeShared(
                const typename Evaluable<InType>::ConstPtr& v) -> Ptr {
                return std::make_shared<NegationEvaluator>(v);
            }

            // ----------------------------------------------------------------------------
            // NegationEvaluator Constructor
            // ----------------------------------------------------------------------------

            template <int DIM>
            NegationEvaluator<DIM>::NegationEvaluator(
                const typename Evaluable<InType>::ConstPtr& v)
                : v_(v) {}

            // ----------------------------------------------------------------------------
            // active
            // ----------------------------------------------------------------------------

            template <int DIM>
            bool NegationEvaluator<DIM>::active() const {
                return v_->active();
            }

            // ----------------------------------------------------------------------------
            // getRelatedVarKeys
            // ----------------------------------------------------------------------------

            template <int DIM>
            void NegationEvaluator<DIM>::getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const {
                v_->getRelatedVarKeys(keys);
            }

            // ----------------------------------------------------------------------------
            // value
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto NegationEvaluator<DIM>::value() const -> OutType {
                return -v_->value();
            }

            // ----------------------------------------------------------------------------
            // forward
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto NegationEvaluator<DIM>::forward() const -> typename Node<OutType>::Ptr {
                const auto child = v_->forward();
                const auto value = -child->value();

                const auto node = Node<OutType>::MakeShared(value);
                node->addChild(child);
                return node;
            }

            // ----------------------------------------------------------------------------
            // backward
            // ----------------------------------------------------------------------------

            template <int DIM>
            void NegationEvaluator<DIM>::backward(
                const Eigen::MatrixXd& lhs, const typename Node<OutType>::Ptr& node,
                StateKeyJacobians& jacs) const {

                const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
                
                if (v_->active()) {
                    v_->backward(-lhs, child, jacs);
                }
            }

            // ----------------------------------------------------------------------------
            // neg
            // ----------------------------------------------------------------------------

            template <int DIM>
            typename NegationEvaluator<DIM>::Ptr neg(
                const typename Evaluable<typename NegationEvaluator<DIM>::InType>::ConstPtr& v) {
                return NegationEvaluator<DIM>::MakeShared(v);
            }

            // Explicit template instantiations
            template class NegationEvaluator<1>;
            template class NegationEvaluator<2>;
            template class NegationEvaluator<3>;

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam
