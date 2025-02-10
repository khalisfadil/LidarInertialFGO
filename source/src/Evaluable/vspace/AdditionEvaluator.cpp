#include "source/include/Evaluable/vspace/AdditionEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // ----------------------------------------------------------------------------
            // MakeShared
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto AdditionEvaluator<DIM>::MakeShared(
                const typename Evaluable<InType>::ConstPtr& v1,
                const typename Evaluable<InType>::ConstPtr& v2) -> Ptr {
                return std::make_shared<AdditionEvaluator>(v1, v2);
            }

            // ----------------------------------------------------------------------------
            // AdditionEvaluator
            // ----------------------------------------------------------------------------

            template <int DIM>
            AdditionEvaluator<DIM>::AdditionEvaluator(
                const typename Evaluable<InType>::ConstPtr& v1,
                const typename Evaluable<InType>::ConstPtr& v2)
                : v1_(v1), v2_(v2) {}

            // ----------------------------------------------------------------------------
            // active
            // ----------------------------------------------------------------------------

            template <int DIM>
            bool AdditionEvaluator<DIM>::active() const {
                return v1_->active() || v2_->active();
            }

            // ----------------------------------------------------------------------------
            // getRelatedVarKeys
            // ----------------------------------------------------------------------------

            template <int DIM>
            void AdditionEvaluator<DIM>::getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const {
                v1_->getRelatedVarKeys(keys);
                v2_->getRelatedVarKeys(keys);
            }


            // ----------------------------------------------------------------------------
            // value
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto AdditionEvaluator<DIM>::value() const -> OutType {
                return v1_->value() + v2_->value();
            }

            // ----------------------------------------------------------------------------
            // forward
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto AdditionEvaluator<DIM>::forward() const -> typename Node<OutType>::Ptr {
                const auto child1 = v1_->forward();
                const auto child2 = v2_->forward();
                const auto value = child1->value() + child2->value();
                const auto node = Node<OutType>::MakeShared(value);
                node->addChild(child1);
                node->addChild(child2);
                return node;
            }

            // ----------------------------------------------------------------------------
            // backward
            // ----------------------------------------------------------------------------
            
            template <int DIM>
            void AdditionEvaluator<DIM>::backward(const Eigen::MatrixXd& lhs,
                                                const typename Node<OutType>::Ptr& node,
                                                StateKeyJacobians& jacs) const {
                const auto child1 = std::static_pointer_cast<Node<InType>>(node->at(0));
                const auto child2 = std::static_pointer_cast<Node<InType>>(node->at(1));

                if (v1_->active()) {
                    v1_->backward(lhs, child1, jacs);
                }

                if (v2_->active()) {
                    v2_->backward(lhs, child2, jacs);
                }
            }

            // ----------------------------------------------------------------------------
            // add
            // ----------------------------------------------------------------------------

            template <int DIM>
            typename AdditionEvaluator<DIM>::Ptr add(
                const typename Evaluable<typename AdditionEvaluator<DIM>::InType>::ConstPtr& v1,
                const typename Evaluable<typename AdditionEvaluator<DIM>::InType>::ConstPtr& v2) {
                return AdditionEvaluator<DIM>::MakeShared(v1, v2);
            }

            // Explicit template instantiations
            template class AdditionEvaluator<1>;
            template class AdditionEvaluator<2>;
            template class AdditionEvaluator<3>;
            
        } // namespace vspace
    }  // namespace eval
}  // namespace slam
