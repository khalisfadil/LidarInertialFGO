#include "source/include/Evaluable/vspace/ScalarMultEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // ----------------------------------------------------------------------------
            // MakeShared
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto ScalarMultEvaluator<DIM>::MakeShared(
                const typename Evaluable<InType>::ConstPtr& v, const double& s) -> Ptr {
                return std::make_shared<ScalarMultEvaluator>(v, s);
            }

            // ----------------------------------------------------------------------------
            // ScalarMultEvaluator Constructor
            // ----------------------------------------------------------------------------

            template <int DIM>
            ScalarMultEvaluator<DIM>::ScalarMultEvaluator(
                const typename Evaluable<InType>::ConstPtr& v, const double& s)
                : v_(v), s_(s) {}

            // ----------------------------------------------------------------------------
            // active
            // ----------------------------------------------------------------------------

            template <int DIM>
            bool ScalarMultEvaluator<DIM>::active() const {
                return v_->active();
            }

            // ----------------------------------------------------------------------------
            // getRelatedVarKeys
            // ----------------------------------------------------------------------------

            template <int DIM>
            void ScalarMultEvaluator<DIM>::getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const {
                v_->getRelatedVarKeys(keys);
            }

            // ----------------------------------------------------------------------------
            // value
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto ScalarMultEvaluator<DIM>::value() const -> OutType {
                return s_ * v_->value();
            }

            // ----------------------------------------------------------------------------
            // forward
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto ScalarMultEvaluator<DIM>::forward() const -> typename Node<OutType>::Ptr {
                const auto child = v_->forward();
                const auto value = s_ * child->value();
                const auto node = Node<OutType>::MakeShared(value);
                node->addChild(child);
                return node;
            }

            // ----------------------------------------------------------------------------
            // backward
            // ----------------------------------------------------------------------------

            template <int DIM>
            void ScalarMultEvaluator<DIM>::backward(const Eigen::MatrixXd& lhs,
                                                    const typename Node<OutType>::Ptr& node,
                                                    StateKeyJacobians& jacs) const {
                const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
                if (v_->active()) {
                    v_->backward(s_ * lhs, child, jacs);
                }
            }

            // ----------------------------------------------------------------------------
            // smult
            // ----------------------------------------------------------------------------

            template <int DIM>
            typename ScalarMultEvaluator<DIM>::Ptr smult(
                const typename Evaluable<typename ScalarMultEvaluator<DIM>::InType>::ConstPtr& v,
                const double& s) {
                return ScalarMultEvaluator<DIM>::MakeShared(v, s);
            }

            // ----------------------------------------------------------------------------
            // Explicit template instantiations
            // ----------------------------------------------------------------------------

            template class ScalarMultEvaluator<1>;
            template class ScalarMultEvaluator<2>;
            template class ScalarMultEvaluator<3>;

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam
