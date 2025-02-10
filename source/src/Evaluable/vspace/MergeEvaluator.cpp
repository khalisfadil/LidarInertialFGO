#include "source/include/Evaluable/vspace/MergeEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // ----------------------------------------------------------------------------
            // MakeShared
            // ----------------------------------------------------------------------------

            template <int DIM1, int DIM2>
            auto MergeEvaluator<DIM1, DIM2>::MakeShared(
                const typename Evaluable<In1Type>::ConstPtr& v1,
                const typename Evaluable<In2Type>::ConstPtr& v2) -> Ptr {
                return std::make_shared<MergeEvaluator>(v1, v2);
            }

            // ----------------------------------------------------------------------------
            // MergeEvaluator Constructor
            // ----------------------------------------------------------------------------

            template <int DIM1, int DIM2>
            MergeEvaluator<DIM1, DIM2>::MergeEvaluator(
                const typename Evaluable<In1Type>::ConstPtr& v1,
                const typename Evaluable<In2Type>::ConstPtr& v2)
                : v1_(v1), v2_(v2) {}

            // ----------------------------------------------------------------------------
            // active
            // ----------------------------------------------------------------------------

            template <int DIM1, int DIM2>
            bool MergeEvaluator<DIM1, DIM2>::active() const {
                return v1_->active() || v2_->active();
            }

            // ----------------------------------------------------------------------------
            // getRelatedVarKeys
            // ----------------------------------------------------------------------------

            template <int DIM1, int DIM2>
            void MergeEvaluator<DIM1, DIM2>::getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const {
                v1_->getRelatedVarKeys(keys);
                v2_->getRelatedVarKeys(keys);
            }

            // ----------------------------------------------------------------------------
            // value
            // ----------------------------------------------------------------------------

            template <int DIM1, int DIM2>
            auto MergeEvaluator<DIM1, DIM2>::value() const -> OutType {
                OutType value;
                value.topRows(DIM1) = v1_->value();
                value.bottomRows(DIM2) = v2_->value();
                return value;
            }

            // ----------------------------------------------------------------------------
            // forward
            // ----------------------------------------------------------------------------

            template <int DIM1, int DIM2>
            auto MergeEvaluator<DIM1, DIM2>::forward() const -> typename Node<OutType>::Ptr {
                const auto child1 = v1_->forward();
                const auto child2 = v2_->forward();

                OutType value;
                value.topRows(DIM1) = child1->value();
                value.bottomRows(DIM2) = child2->value();

                const auto node = Node<OutType>::MakeShared(value);
                node->addChild(child1);
                node->addChild(child2);

                return node;
            }

            // ----------------------------------------------------------------------------
            // backward
            // ----------------------------------------------------------------------------

            template <int DIM1, int DIM2>
            void MergeEvaluator<DIM1, DIM2>::backward(
                const Eigen::MatrixXd& lhs, const typename Node<OutType>::Ptr& node,
                StateKeyJacobians& jacs) const {
                
                if (v1_->active()) {
                    const auto child1 = std::static_pointer_cast<Node<In1Type>>(node->at(0));
                    v1_->backward(lhs.topRows(DIM1), child1, jacs);
                }

                if (v2_->active()) {
                    const auto child2 = std::static_pointer_cast<Node<In2Type>>(node->at(1));
                    v2_->backward(lhs.bottomRows(DIM2), child2, jacs);
                }
            }

            // ----------------------------------------------------------------------------
            // merge
            // ----------------------------------------------------------------------------

            template <int DIM1, int DIM2>
            typename MergeEvaluator<DIM1, DIM2>::Ptr merge(
                const typename Evaluable<typename MergeEvaluator<DIM1, DIM2>::In1Type>::ConstPtr& v1,
                const typename Evaluable<typename MergeEvaluator<DIM1, DIM2>::In2Type>::ConstPtr& v2) {
                return MergeEvaluator<DIM1, DIM2>::MakeShared(v1, v2);
            }

            // Explicit template instantiations
            template class MergeEvaluator<1, 1>;
            template class MergeEvaluator<2, 2>;
            template class MergeEvaluator<3, 3>;

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam
