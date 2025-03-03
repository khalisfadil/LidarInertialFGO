#include "source/include/Evaluable/point2point/YawVelErrorEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace p2p {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------
            auto YawVelErrorEvaluator::MakeShared(const Eigen::Matrix<double, 1, 1>& vel_meas,
                                                const Evaluable<InType>::ConstPtr& w_iv_inv) -> Ptr {
                return std::make_shared<YawVelErrorEvaluator>(vel_meas, w_iv_inv);
            }

            // -----------------------------------------------------------------------------
            // Constructor
            // -----------------------------------------------------------------------------
            YawVelErrorEvaluator::YawVelErrorEvaluator(const Eigen::Matrix<double, 1, 1>& vel_meas,
                                                    const Evaluable<InType>::ConstPtr& w_iv_inv)
                : vel_meas_(vel_meas), 
                w_iv_inv_(w_iv_inv),
                D_((Eigen::Matrix<double, 1, 6>() << 0, 0, 0, 0, 0, 1).finished()) {}

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------
            bool YawVelErrorEvaluator::active() const { 
                return w_iv_inv_->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------
            void YawVelErrorEvaluator::getRelatedVarKeys(KeySet& keys) const { 
                w_iv_inv_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------
            auto YawVelErrorEvaluator::value() const -> OutType { 
                return vel_meas_ - D_ * w_iv_inv_->value();
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------
            auto YawVelErrorEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child = w_iv_inv_->forward();
                auto node = Node<OutType>::MakeShared(vel_meas_ - D_ * child->value());
                node->addChild(child);
                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------
            void YawVelErrorEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                                const Node<OutType>::Ptr& node,
                                                StateKeyJacobians& jacs) const {
                if (w_iv_inv_->active()) {
                    w_iv_inv_->backward(lhs * -D_, std::static_pointer_cast<Node<InType>>(node->getChild(0)), jacs);
                }
            }

            // -----------------------------------------------------------------------------
            // velError
            // -----------------------------------------------------------------------------
            auto velError(const Eigen::Matrix<double, 1, 1>& vel_meas,
                        const Evaluable<YawVelErrorEvaluator::InType>::ConstPtr& w_iv_inv) -> YawVelErrorEvaluator::Ptr {
                return YawVelErrorEvaluator::MakeShared(vel_meas, w_iv_inv);
            }

        }  // namespace p2p
    }  // namespace eval
}  // namespace slam
