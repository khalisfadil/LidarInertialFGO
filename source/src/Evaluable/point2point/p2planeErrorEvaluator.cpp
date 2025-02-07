#include "source/include/Evaluable/point2point/p2planeErrorEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace p2p {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto P2PlaneErrorEvaluator::MakeShared(
                const Evaluable<InType>::ConstPtr& T_rq,
                const Eigen::Vector3d& reference,
                const Eigen::Vector3d& query,
                const Eigen::Vector3d& normal) -> Ptr {
                return std::make_shared<P2PlaneErrorEvaluator>(T_rq, reference, query, normal);
            }

            // -----------------------------------------------------------------------------
            // P2PlaneErrorEvaluator
            // -----------------------------------------------------------------------------

            P2PlaneErrorEvaluator::P2PlaneErrorEvaluator(
                const Evaluable<InType>::ConstPtr& T_rq,
                const Eigen::Vector3d& reference,
                const Eigen::Vector3d& query,
                const Eigen::Vector3d& normal)
                : T_rq_(T_rq), reference_(reference), query_(query), normal_(normal) {}

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------

            bool P2PlaneErrorEvaluator::active() const { return T_rq_->active(); }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void P2PlaneErrorEvaluator::getRelatedVarKeys(KeySet& keys) const {
                T_rq_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto P2PlaneErrorEvaluator::value() const -> OutType {
                // Ensure the transformation is valid
                if (!T_rq_) {
                    throw std::runtime_error("[P2PlaneErrorEvaluator::value] T_rq_ is nullptr.");
                }

                // Retrieve SE(3) transformation matrix
                const Eigen::Matrix4d& T_rq = T_rq_->value().matrix();
                
                // Extract rotation and translation for efficient computation
                const Eigen::Matrix3d& R_rq = T_rq.block<3, 3>(0, 0);
                const Eigen::Vector3d& t_rq = T_rq.block<3, 1>(0, 3);

                // Compute transformed query point
                const Eigen::Vector3d transformed_query = R_rq * query_ + t_rq;

                // Compute point-to-plane error and convert it to Eigen::Matrix<1,1>
                OutType error;
                error(0, 0) = normal_.dot(reference_ - transformed_query);
                return error;
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------
            
            auto P2PlaneErrorEvaluator::forward() const -> Node<OutType>::Ptr {
                // Retrieve child node safely
                const auto child = T_rq_->forward();
                if (!child) {
                    throw std::runtime_error("[P2PlaneErrorEvaluator::forward] Null child node encountered.");
                }

                // Compute transformed query point
                const Eigen::Matrix4d& T_rq = child->value().matrix();
                const Eigen::Vector3d transformed_query = T_rq.block<3, 3>(0, 0) * query_ + T_rq.block<3, 1>(0, 3);

                // Compute error
                OutType error = normal_.transpose() * (reference_ - transformed_query);

                // Create new node and attach dependencies
                const auto node = Node<OutType>::MakeShared(error);
                node->addChild(child);

                return node;
            }


            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void P2PlaneErrorEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                     const Node<OutType>::Ptr& node,
                                     StateKeyJacobians& jacs) const {
                if (!node || node->size() < 1) {
                    throw std::runtime_error("[P2PlaneErrorEvaluator::backward] Node has insufficient children.");
                }

                // Retrieve child node safely
                auto child_base = node->getChild(0);
                if (!child_base) {
                    throw std::runtime_error("[P2PlaneErrorEvaluator::backward] Null child node encountered.");
                }

                // Attempt to cast to the correct type
                auto child = std::dynamic_pointer_cast<Node<InType>>(child_base);
                if (!child || !child->hasValue()) {
                    throw std::runtime_error("[P2PlaneErrorEvaluator::backward] Invalid child node.");
                }

                if (T_rq_->active()) {
                    const Eigen::Matrix4d& T_rq = child->value().matrix();
                    const Eigen::Vector3d transformed_query = T_rq.block<3, 3>(0, 0) * query_ + T_rq.block<3, 1>(0, 3);

                    // Compute new LHS Jacobian
                    Eigen::Matrix<double, 1, 6> new_lhs = -lhs * normal_.transpose() *
                                                        slam::liemath::se3::point2fs(transformed_query).block<3, 6>(0, 0);

                    // Propagate gradients
                    T_rq_->backward(new_lhs, child, jacs);
                }
            }

            // -----------------------------------------------------------------------------
            // getJacobianPose
            // -----------------------------------------------------------------------------
            
            Eigen::Matrix<double, 1, 6> P2PlaneErrorEvaluator::getJacobianPose() const {
                // Ensure transform is valid
                if (!T_rq_) {
                    throw std::runtime_error("[P2PlaneErrorEvaluator::getJacobianPose] T_rq_ is nullptr.");
                }

                // Retrieve SE(3) transformation matrix
                const Eigen::Matrix4d& T_rq = T_rq_->value().matrix();
                
                // Compute transformed query point in world frame
                const Eigen::Vector3d transformed_query = T_rq.block<3, 3>(0, 0) * query_ + T_rq.block<3, 1>(0, 3);
                
                // Compute and return Jacobian
                return -normal_.transpose() * slam::liemath::se3::point2fs(transformed_query).block<3, 6>(0, 0);
            }

            // -----------------------------------------------------------------------------
            // p2planeError
            // -----------------------------------------------------------------------------
            
            P2PlaneErrorEvaluator::Ptr p2planeError(
                const Evaluable<P2PlaneErrorEvaluator::InType>::ConstPtr& T_rq,
                const Eigen::Vector3d& reference, 
                const Eigen::Vector3d& query, 
                const Eigen::Vector3d& normal) {
                return P2PlaneErrorEvaluator::MakeShared(T_rq, reference, query, normal);
            }

        }  // namespace p2p
    }  // namespace eval
}  // namespace slam
