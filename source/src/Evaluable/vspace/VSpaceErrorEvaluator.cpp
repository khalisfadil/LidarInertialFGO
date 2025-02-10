#include "source/include/Evaluable/vspace/VSpaceErrorEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // ----------------------------------------------------------------------------
            // MakeShared
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto VSpaceErrorEvaluator<DIM>::MakeShared(
                const typename Evaluable<InType>::ConstPtr& v, const InType& v_meas) -> Ptr {
                return std::make_shared<VSpaceErrorEvaluator>(v, v_meas);
            }

            // ----------------------------------------------------------------------------
            // VSpaceErrorEvaluator Constructor
            // ----------------------------------------------------------------------------

            template <int DIM>
            VSpaceErrorEvaluator<DIM>::VSpaceErrorEvaluator(
                const typename Evaluable<InType>::ConstPtr& v, const InType& v_meas)
                : v_(v), v_meas_(v_meas) {}

            // ----------------------------------------------------------------------------
            // active
            // ----------------------------------------------------------------------------

            template <int DIM>
            bool VSpaceErrorEvaluator<DIM>::active() const {
                return v_->active();
            }

            // ----------------------------------------------------------------------------
            // getRelatedVarKeys
            // ----------------------------------------------------------------------------

            template <int DIM>
            void VSpaceErrorEvaluator<DIM>::getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const {
                v_->getRelatedVarKeys(keys);
            }

            // ----------------------------------------------------------------------------
            // value
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto VSpaceErrorEvaluator<DIM>::value() const -> OutType {
                return v_meas_ - v_->value();
            }

            // ----------------------------------------------------------------------------
            // forward
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto VSpaceErrorEvaluator<DIM>::forward() const -> typename Node<OutType>::Ptr {
                const auto child = v_->forward();
                const auto value = v_meas_ - child->value();
                const auto node = Node<OutType>::MakeShared(value);
                node->addChild(child);
                return node;
            }

            // ----------------------------------------------------------------------------
            // backward
            // ----------------------------------------------------------------------------

            template <int DIM>
            void VSpaceErrorEvaluator<DIM>::backward(const Eigen::MatrixXd& lhs,
                                                     const typename Node<OutType>::Ptr& node,
                                                     StateKeyJacobians& jacs) const {
                if (v_->active()) {
                    const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
                    v_->backward(-lhs, child, jacs);
                }
            }

            // ----------------------------------------------------------------------------
            // vspace_error
            // ----------------------------------------------------------------------------

            template <int DIM>
            typename VSpaceErrorEvaluator<DIM>::Ptr vspace_error(
                const typename Evaluable<typename VSpaceErrorEvaluator<DIM>::InType>::ConstPtr& v,
                const typename VSpaceErrorEvaluator<DIM>::InType& v_meas) {
                return VSpaceErrorEvaluator<DIM>::MakeShared(v, v_meas);
            }

            // ----------------------------------------------------------------------------
            // Explicit template instantiations
            // ----------------------------------------------------------------------------

            template class VSpaceErrorEvaluator<1>;
            template class VSpaceErrorEvaluator<2>;
            template class VSpaceErrorEvaluator<3>;

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam
