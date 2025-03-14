#include "Core/Evaluable/point2point/p2pErrorEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace p2p {
            
            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto P2PErrorEvaluator::MakeShared(const Evaluable<InType>::ConstPtr& T_rq,
                                               const Eigen::Vector3d& reference,
                                               const Eigen::Vector3d& query) -> Ptr {
                return std::make_shared<P2PErrorEvaluator>(T_rq, reference, query);
            }

            // -----------------------------------------------------------------------------
            // P2PErrorEvaluator
            // -----------------------------------------------------------------------------
            
            P2PErrorEvaluator::P2PErrorEvaluator(const Evaluable<InType>::ConstPtr& T_rq,
                                                 const Eigen::Vector3d& reference,
                                                 const Eigen::Vector3d& query)
                : T_rq_(T_rq) {
                D_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
                reference_.block<3, 1>(0, 0) = reference;
                query_.block<3, 1>(0, 0) = query;
            }

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------

            bool P2PErrorEvaluator::active() const { return T_rq_->active(); }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void P2PErrorEvaluator::getRelatedVarKeys(KeySet& keys) const {
                T_rq_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto P2PErrorEvaluator::value() const -> OutType {
                return D_ * (reference_ - T_rq_->value() * query_);
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto P2PErrorEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child = T_rq_->forward();
                if (!child) {
                    throw std::runtime_error("[P2PErrorEvaluator::forward] Null child node.");
                }

                const auto& T_rq = child->value();
                OutType error;
                error.noalias() = D_ * (reference_ - T_rq * query_); // Use noalias() to optimize matrix operations

                auto node = Node<OutType>::MakeShared(error);
                node->addChild(child);
                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void P2PErrorEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                 const Node<OutType>::Ptr& node,
                                 StateKeyJacobians& jacs) const {
                if (!node || node->size() < 1) {
                    throw std::runtime_error("[P2PErrorEvaluator::backward] Node has insufficient children.");
                }

                auto child_base = node->getChild(0);
                if (!child_base) {
                    throw std::runtime_error("[P2PErrorEvaluator::backward] Null child node encountered.");
                }

                auto child = std::dynamic_pointer_cast<Node<InType>>(child_base);
                if (!child || !child->hasValue()) {
                    throw std::runtime_error("[P2PErrorEvaluator::backward] Invalid child node.");
                }

                if (T_rq_->active()) {
                    const auto& T_rq = child->value();
                    Eigen::Matrix<double, 3, 1> Tq = (T_rq * query_).template block<3, 1>(0, 0);

                    Eigen::Matrix<double, 3, 6> new_lhs;
                    new_lhs.noalias() = -lhs * D_ * slam::liemath::se3::point2fs(Tq);

                    T_rq_->backward(new_lhs, child, jacs);
                }
            }

            // -----------------------------------------------------------------------------
            // getJacobianPose
            // -----------------------------------------------------------------------------

            Eigen::Matrix<double, 3, 6> P2PErrorEvaluator::getJacobianPose() const {
                const auto& T_rq = T_rq_->value();
                Eigen::Matrix<double, 3, 1> Tq = (T_rq * query_).template block<3, 1>(0, 0);

                Eigen::Matrix<double, 3, 6> jac;
                jac.noalias() = -D_ * slam::liemath::se3::point2fs(Tq);
                return jac;
            }

            // -----------------------------------------------------------------------------
            // p2pError
            // -----------------------------------------------------------------------------

            P2PErrorEvaluator::Ptr p2pError(
                const Evaluable<P2PErrorEvaluator::InType>::ConstPtr& T_rq,
                const Eigen::Vector3d& reference, const Eigen::Vector3d& query) {
                return P2PErrorEvaluator::MakeShared(T_rq, reference, query);
            }

        }  // namespace p2p
    }  // namespace eval
}  // namespace slam
