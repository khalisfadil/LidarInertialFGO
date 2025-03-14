#include "Core/Evaluable/point2point/p2planeGlobalPerturbEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace p2p {
            
            // -----------------------------------------------------------------------------
            // Factory function to create a shared instance of P2PlaneErrorGlobalPerturbEvaluator.
            // -----------------------------------------------------------------------------

            auto P2PlaneErrorGlobalPerturbEvaluator::MakeShared(
                const Evaluable<InType>::ConstPtr &T_rq,
                const Eigen::Vector3d &reference,
                const Eigen::Vector3d &query,
                const Eigen::Vector3d &normal) -> Ptr {
                return std::make_shared<P2PlaneErrorGlobalPerturbEvaluator>(T_rq, reference, query, normal);
            }

            // -----------------------------------------------------------------------------
            // Constructor for P2PlaneErrorGlobalPerturbEvaluator.
            // -----------------------------------------------------------------------------

            P2PlaneErrorGlobalPerturbEvaluator::P2PlaneErrorGlobalPerturbEvaluator(
                const Evaluable<InType>::ConstPtr &T_rq,
                const Eigen::Vector3d &reference,
                const Eigen::Vector3d &query,
                const Eigen::Vector3d &normal)
                : T_rq_(T_rq), reference_(reference), query_(query), normal_(normal) {}

            // -----------------------------------------------------------------------------
            // Check if this evaluator is dependent on active state variables.
            // -----------------------------------------------------------------------------

            bool P2PlaneErrorGlobalPerturbEvaluator::active() const {
                return T_rq_->active();
            }

            // -----------------------------------------------------------------------------
            // Collects all related variable keys.
            // -----------------------------------------------------------------------------
 
            void P2PlaneErrorGlobalPerturbEvaluator::getRelatedVarKeys(KeySet &keys) const {
                T_rq_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // Computes the point-to-plane error.
            // -----------------------------------------------------------------------------

            auto P2PlaneErrorGlobalPerturbEvaluator::value() const -> OutType {
                // Ensure the transformation is valid
                if (!T_rq_) {
                    throw std::runtime_error("[P2PlaneErrorGlobalPerturbEvaluator::value] T_rq_ is nullptr.");
                }

                // Retrieve SE(3) transformation components
                const Eigen::Matrix4d& T_rq = T_rq_->value().matrix();
                const Eigen::Matrix3d& R_rq = T_rq.block<3, 3>(0, 0);
                const Eigen::Vector3d& t_rq = T_rq.block<3, 1>(0, 3);

                // Compute transformed query point
                const Eigen::Vector3d transformed_query = R_rq * query_ + t_rq;

                // Compute signed point-to-plane error
                OutType error;
                error(0, 0) = normal_.dot(reference_ - transformed_query);
                return error;
            }

            // -----------------------------------------------------------------------------
            // Forward pass for automatic differentiation.
            // -----------------------------------------------------------------------------

            auto P2PlaneErrorGlobalPerturbEvaluator::forward() const -> Node<OutType>::Ptr {
                // Retrieve the transformation node safely
                const auto child = T_rq_->forward();
                if (!child) {
                    throw std::runtime_error(
                        "[P2PlaneErrorGlobalPerturbEvaluator::forward] Null child node.");
                }
            
                // Compute error using cached transformation
                const auto T_rq = child->value().matrix();
                OutType error = normal_.transpose() *
                                (reference_ - T_rq.block<3, 3>(0, 0) * query_ -
                                T_rq.block<3, 1>(0, 3));

                // Create a new node and attach dependencies
                const auto node = Node<OutType>::MakeShared(error);
                node->addChild(child);
                return node;
            }

            // -----------------------------------------------------------------------------
            // Backward pass for Jacobian computation.
            // -----------------------------------------------------------------------------

            void P2PlaneErrorGlobalPerturbEvaluator::backward(
                const Eigen::Ref<const Eigen::MatrixXd> &lhs,
                const Node<OutType>::Ptr &node, StateKeyJacobians &jacs) const {
                if (!node || node->size() < 1) {
                    throw std::runtime_error(
                        "[P2PlaneErrorGlobalPerturbEvaluator::backward] Node has insufficient "
                        "children.");
                }

                // Retrieve child node safely
                auto child_base = node->getChild(0);
                if (!child_base) {
                    throw std::runtime_error(
                        "[P2PlaneErrorGlobalPerturbEvaluator::backward] Null child node "
                        "encountered.");
                }

                // Attempt to cast to correct type
                auto child = std::dynamic_pointer_cast<Node<InType>>(child_base);
                if (!child || !child->hasValue()) {
                    throw std::runtime_error(
                        "[P2PlaneErrorGlobalPerturbEvaluator::backward] Invalid child node.");
                }

                if (T_rq_->active()) {
                    const Eigen::Matrix4d T_rq = child->value().matrix();
                    const Eigen::Matrix3d C_rq = T_rq.block<3, 3>(0, 0);

                    // Preallocate Jacobian matrix
                    Eigen::Matrix<double, 1, 6> jac = Eigen::Matrix<double, 1, 6>::Zero();

                    // Compute Jacobian components
                    jac.block<1, 3>(0, 0) = -normal_.transpose() * C_rq;
                    jac.block<1, 3>(0, 3) = normal_.transpose() * C_rq * slam::liemath::so3::hat(query_);

                    // Propagate gradients
                    T_rq_->backward(lhs * jac, child, jacs);
                }
            }

            // -----------------------------------------------------------------------------
            // Computes the Jacobian of the point-to-plane error w.r.t pose.
            // -----------------------------------------------------------------------------

            Eigen::Matrix<double, 1, 6> P2PlaneErrorGlobalPerturbEvaluator::getJacobianPose() const {
                // Ensure the transformation is valid
                if (!T_rq_) {
                    throw std::runtime_error("[P2PlaneErrorGlobalPerturbEvaluator::getJacobianPose] T_rq_ is nullptr.");
                }

                // Retrieve SE(3) transformation matrix
                const Eigen::Matrix4d& T_rq = T_rq_->value().matrix();
                const Eigen::Matrix3d& R_rq = T_rq.block<3, 3>(0, 0);

                // Preallocate Jacobian matrix
                Eigen::Matrix<double, 1, 6> jac;
                
                // Compute Jacobian components
                jac.template leftCols<3>() = -normal_.transpose();  // Translation part
                jac.template rightCols<3>() = normal_.transpose() * R_rq * slam::liemath::so3::hat(query_); // Rotation part

                return jac;
            }

            // -----------------------------------------------------------------------------
            // Factory function to create an instance of P2PlaneErrorGlobalPerturbEvaluator.
            // -----------------------------------------------------------------------------

            P2PlaneErrorGlobalPerturbEvaluator::Ptr p2planeGlobalError(
                const Evaluable<P2PlaneErrorGlobalPerturbEvaluator::InType>::ConstPtr &T_rq,
                const Eigen::Vector3d &reference, const Eigen::Vector3d &query,
                const Eigen::Vector3d &normal) {
                return P2PlaneErrorGlobalPerturbEvaluator::MakeShared(T_rq, reference, query,
                                                                    normal);
            }

        }  // namespace p2p
    } // namespace eval
}  // namespace slam
