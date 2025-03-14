#include "Core/Evaluable/point2point/YawErrorEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace p2p {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto YawErrorEvaluator::MakeShared(double yaw_meas, const Evaluable<PoseInType>::ConstPtr &T_ms_prev, 
                                    const Evaluable<PoseInType>::ConstPtr &T_ms_curr) -> Ptr {
                return std::make_shared<YawErrorEvaluator>(yaw_meas, T_ms_prev, T_ms_curr);
            }

            // -----------------------------------------------------------------------------
            // YawErrorEvaluator
            // -----------------------------------------------------------------------------

            YawErrorEvaluator::YawErrorEvaluator(double yaw_meas, 
                                const Evaluable<PoseInType>::ConstPtr &T_ms_prev, 
                                const Evaluable<PoseInType>::ConstPtr &T_ms_curr)
            : yaw_meas_(yaw_meas), T_ms_prev_(T_ms_prev), T_ms_curr_(T_ms_curr) {
                d_ = Eigen::Matrix<double, 1, 3>::Zero();
                d_(0, 2) = 1.0;
            }
            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------

            bool YawErrorEvaluator::active() const { 
                return T_ms_prev_->active() || T_ms_curr_->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void YawErrorEvaluator::getRelatedVarKeys(KeySet& keys) const { 
                T_ms_prev_->getRelatedVarKeys(keys);
                T_ms_curr_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto YawErrorEvaluator::value() const -> OutType { 
                // Form measured and predicted printegrated DCM: prev (p) curr (c)
                Eigen::Vector3d meas_vec(0.0, 0.0, yaw_meas_);
                const liemath::so3::Rotation C_pc_meas(meas_vec);
                const liemath::so3::Rotation C_cp_eval((T_ms_curr_->value().C_ba().inverse() * 
                                                                T_ms_prev_->value().C_ba()).eval());
                // Return error
                return d_ * liemath::so3::rot2vec((C_cp_eval * C_pc_meas).matrix());
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto YawErrorEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child1 = T_ms_prev_->forward();
                const auto child2 = T_ms_curr_->forward();

                const auto C_ms_prev = child1->value().C_ba();
                const auto C_ms_curr = child2->value().C_ba();

                Eigen::Vector3d meas_vec(0.0, 0.0, yaw_meas_);
                const liemath::so3::Rotation C_pc_meas(meas_vec);
                const liemath::so3::Rotation C_cp_eval((C_ms_curr.transpose() * C_ms_prev).eval());
                OutType error = d_ * liemath::so3::rot2vec((C_cp_eval * C_pc_meas).matrix());
                // clang-format on

                const auto node = Node<OutType>::MakeShared(error);
                node->addChild(child1);
                node->addChild(child2);
                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void YawErrorEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                 const Node<OutType>::Ptr& node,
                                 StateKeyJacobians& jacs) const {
                constexpr double jacobian_value = 1.0;
                const Eigen::Matrix<double, 1, 6> jac = (Eigen::Matrix<double, 1, 6>() << 0, 0, 0, 0, 0, jacobian_value).finished();

                const auto child1 = std::static_pointer_cast<Node<PoseInType>>(node->getChild(0));
                const auto child2 = std::static_pointer_cast<Node<PoseInType>>(node->getChild(1));

                if (T_ms_prev_->active()) {
                    T_ms_prev_->backward(lhs * -jac, child1, jacs);
                }
                if (T_ms_curr_->active()) {
                    T_ms_curr_->backward(lhs * jac, child2, jacs);
                }
            }


            // -----------------------------------------------------------------------------
            // velError
            // -----------------------------------------------------------------------------

            YawErrorEvaluator::Ptr velError(const double yaw_meas,
                                            const Evaluable<YawErrorEvaluator::PoseInType>::ConstPtr &T_ms_prev,
                                            const Evaluable<YawErrorEvaluator::PoseInType>::ConstPtr &T_ms_curr) {
                return YawErrorEvaluator::MakeShared(yaw_meas, T_ms_prev, T_ms_curr);
            }

        }  // namespace p2p
    }  // namespace eval
}  // namespace slam