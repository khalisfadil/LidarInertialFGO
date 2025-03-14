#include "Core/Evaluable/se3/PoseInterpolator.hpp"

namespace slam {
    namespace eval {
        namespace se3 {
            
            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            PoseInterpolator::Ptr PoseInterpolator::MakeShared(
                const Time& time,
                const Evaluable<InType>::ConstPtr& transform1,
                const Time& time1,
                const Evaluable<InType>::ConstPtr& transform2,
                const Time& time2) {
                return std::make_shared<PoseInterpolator>(time, transform1, time1, transform2, time2);
            }

            // -----------------------------------------------------------------------------
            // PoseInterpolator
            // -----------------------------------------------------------------------------

            PoseInterpolator::PoseInterpolator(
                const Time& time,
                const Evaluable<InType>::ConstPtr& transform1,
                const Time& time1,
                const Evaluable<InType>::ConstPtr& transform2,
                const Time& time2)
                : transform1_(transform1), transform2_(transform2) {

                alpha_ = (time - time1).seconds() / (time2 - time1).seconds();

                // **Precompute Faulhaber coefficients (now in a fixed array)**
                faulhaber_coeffs_ = {alpha_,
                                    alpha_ * (alpha_ - 1) / 2,
                                    alpha_ * (alpha_ - 1) * (2 * alpha_ - 1) / 12,
                                    alpha_ * alpha_ * (alpha_ - 1) * (alpha_ - 1) / 24};
            }

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------

            bool PoseInterpolator::active() const {
                return transform1_->active() || transform2_->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void PoseInterpolator::getRelatedVarKeys(KeySet& keys) const {
                transform1_->getRelatedVarKeys(keys);
                transform2_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto PoseInterpolator::value() const -> OutType {
                const auto& T1 = transform1_->value();
                const auto& T2 = transform2_->value();
                
                // Explicitly cast xi_i1 to Eigen::VectorXd to match the constructor
                Eigen::Matrix<double, 6, 1> xi_i1 = alpha_ * (T2 / T1).vec();
                Eigen::Ref<const Eigen::VectorXd> xi_ref(xi_i1);  // **Explicit reference**
                
                return slam::liemath::se3::Transformation(xi_ref) * T1;
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto PoseInterpolator::forward() const -> Node<OutType>::Ptr {
                const auto& T1 = transform1_->forward();
                const auto& T2 = transform2_->forward();

                Eigen::Matrix<double, 6, 1> xi_i1 = alpha_ * (T2->value() / T1->value()).vec();
                Eigen::Ref<const Eigen::VectorXd> xi_ref(xi_i1);  // **Explicit cast**

                const slam::liemath::se3::Transformation T_i1(xi_ref);
                OutType T_i0 = T_i1 * T1->value();

                // **First create the node**
                auto node = Node<OutType>::MakeShared(T_i0);

                // **Then add children separately**
                node->addChild(T1);
                node->addChild(T2);

                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void PoseInterpolator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs, 
                                            const Node<OutType>::Ptr& node, 
                                            StateKeyJacobians& jacs) const {
                if (!active()) return;

                const auto& T1 = transform1_->value();
                const auto& T2 = transform2_->value();

                // Compute the Lie algebra representation of relative transformation
                Eigen::Matrix<double, 6, 6> xi_21_curlyhat = slam::liemath::se3::curlyhat((T2 / T1).vec());

                // Compute Faulhaber series expansion for Jacobian approximation
                Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
                Eigen::Matrix<double, 6, 6> xictmp = Eigen::Matrix<double, 6, 6>::Identity();

                for (size_t i = 0; i < faulhaber_coeffs_.size(); ++i) {
                    if (i > 0) xictmp = xi_21_curlyhat * xictmp;
                    A.noalias() += faulhaber_coeffs_[i] * xictmp;  // **Optimized for memory efficiency**
                }

                // **Use getChild() instead of at()**
                if (transform1_->active()) {
                    const auto T1_ = std::static_pointer_cast<Node<InType>>(node->getChild(0));
                    transform1_->backward(lhs * (Eigen::Matrix<double, 6, 6>::Identity() - A), T1_, jacs);
                }

                if (transform2_->active()) {
                    const auto T2_ = std::static_pointer_cast<Node<InType>>(node->getChild(1));
                    transform2_->backward(lhs * A, T2_, jacs);
                }
            }
        }  // namespace se3
    }  // namespace eval
}  // namespace slam
