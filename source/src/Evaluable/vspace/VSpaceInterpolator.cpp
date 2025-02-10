#include "source/include/Evaluable/vspace/VSpaceInterpolator.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // ----------------------------------------------------------------------------
            // MakeShared
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto VSpaceInterpolator<DIM>::MakeShared(
                const Time& time, 
                const typename Evaluable<InType>::ConstPtr& bias1, 
                const Time& time1, 
                const typename Evaluable<InType>::ConstPtr& bias2, 
                const Time& time2) -> Ptr {
                return std::make_shared<VSpaceInterpolator>(time, bias1, time1, bias2, time2);
            }

            // ----------------------------------------------------------------------------
            // VSpaceInterpolator Constructor
            // ----------------------------------------------------------------------------

            template <int DIM>
            VSpaceInterpolator<DIM>::VSpaceInterpolator(
                const Time& time, 
                const typename Evaluable<InType>::ConstPtr& bias1, 
                const Time& time1, 
                const typename Evaluable<InType>::ConstPtr& bias2, 
                const Time& time2)
                : bias1_(bias1), bias2_(bias2) {

                // Ensure the interpolation time is within valid range
                if (time < time1 || time > time2) {
                    throw std::runtime_error("[VSpaceInterpolator] Interpolation time out of range.");
                }

                // Compute interpolation weights
                const double tau = (time - time1).seconds();
                const double T = (time2 - time1).seconds();
                const double ratio = tau / T;
                
                psi_ = ratio;
                lambda_ = 1.0 - ratio;
            }

            // ----------------------------------------------------------------------------
            // active
            // ----------------------------------------------------------------------------

            template <int DIM>
            bool VSpaceInterpolator<DIM>::active() const {
                return bias1_->active() || bias2_->active();
            }

            // ----------------------------------------------------------------------------
            // getRelatedVarKeys
            // ----------------------------------------------------------------------------

            template <int DIM>
            void VSpaceInterpolator<DIM>::getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const {
                bias1_->getRelatedVarKeys(keys);
                bias2_->getRelatedVarKeys(keys);
            }

            // ----------------------------------------------------------------------------
            // value
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto VSpaceInterpolator<DIM>::value() const -> OutType {
                return lambda_ * bias1_->value() + psi_ * bias2_->value();
            }

            // ----------------------------------------------------------------------------
            // forward
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto VSpaceInterpolator<DIM>::forward() const -> typename Node<OutType>::Ptr {
                const auto b1 = bias1_->forward();
                const auto b2 = bias2_->forward();

                OutType interpolated_value = lambda_ * b1->value() + psi_ * b2->value();

                const auto node = Node<OutType>::MakeShared(interpolated_value);
                node->addChild(b1);
                node->addChild(b2);
                
                return node;
            }

            // ----------------------------------------------------------------------------
            // backward
            // ----------------------------------------------------------------------------

            template <int DIM>
            void VSpaceInterpolator<DIM>::backward(const Eigen::MatrixXd& lhs,
                                                   const typename Node<OutType>::Ptr& node,
                                                   StateKeyJacobians& jacs) const {
                if (!active()) return;

                if (bias1_->active()) {
                    const auto b1_ = std::static_pointer_cast<Node<InType>>(node->at(0));
                    bias1_->backward(lambda_ * lhs, b1_, jacs);
                }

                if (bias2_->active()) {
                    const auto b2_ = std::static_pointer_cast<Node<InType>>(node->at(1));
                    bias2_->backward(psi_ * lhs, b2_, jacs);
                }
            }

            // ----------------------------------------------------------------------------
            // Explicit template instantiations
            // ----------------------------------------------------------------------------

            template class VSpaceInterpolator<1>;
            template class VSpaceInterpolator<2>;
            template class VSpaceInterpolator<3>;

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam
