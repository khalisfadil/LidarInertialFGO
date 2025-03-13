#include "Evaluable/point2point/VelErrorEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace p2p {

            // -----------------------------------------------------------------------------
            // Factory function to create a shared instance of P2PlaneErrorGlobalPerturbEvaluator.
            // -----------------------------------------------------------------------------

            auto VelErrorEvaluator::MakeShared(const Eigen::Vector2d& vel_meas, const Evaluable<InType>::ConstPtr& w_iv_inv) -> Ptr {
                return std::make_shared<VelErrorEvaluator>(vel_meas, w_iv_inv);
            }

            // -----------------------------------------------------------------------------
            // Factory function to create a shared instance of P2PlaneErrorGlobalPerturbEvaluator.
            // -----------------------------------------------------------------------------

            VelErrorEvaluator::VelErrorEvaluator(const Eigen::Vector2d& vel_meas, 
                                     const Evaluable<InType>::ConstPtr &w_iv_inv)
            : vel_meas_(vel_meas), w_iv_inv_(w_iv_inv),
            D_((Eigen::Matrix<double, 2, 6>() << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 1.0, 0.0, 0.0, 0.0, 0.0).finished()) {}

            // -----------------------------------------------------------------------------
            // Factory function to create a shared instance of P2PlaneErrorGlobalPerturbEvaluator.
            // -----------------------------------------------------------------------------

            bool VelErrorEvaluator::active() const { 
                return w_iv_inv_ && w_iv_inv_->active();
            }

            // -----------------------------------------------------------------------------
            // Factory function to create a shared instance of P2PlaneErrorGlobalPerturbEvaluator.
            // -----------------------------------------------------------------------------

            void VelErrorEvaluator::getRelatedVarKeys(KeySet& keys) const { 
                if (w_iv_inv_) w_iv_inv_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // Factory function to create a shared instance of P2PlaneErrorGlobalPerturbEvaluator.
            // -----------------------------------------------------------------------------

            auto VelErrorEvaluator::value() const -> OutType { 
                return vel_meas_ - D_ * w_iv_inv_->value();
            }

            // -----------------------------------------------------------------------------
            // Factory function to create a shared instance of P2PlaneErrorGlobalPerturbEvaluator.
            // -----------------------------------------------------------------------------

            auto VelErrorEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child1 = w_iv_inv_->forward();
                // clang-format off
                OutType error = vel_meas_ - D_ * child1->value();
                // clang-format on

                const auto node = Node<OutType>::MakeShared(error);
                node->addChild(child1);
                return node;
            }

            // -----------------------------------------------------------------------------
            // Factory function to create a shared instance of P2PlaneErrorGlobalPerturbEvaluator.
            // -----------------------------------------------------------------------------

            void VelErrorEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                  const Node<OutType>::Ptr& node,
                                  StateKeyJacobians& jacs) const {
                if (w_iv_inv_->active()) {
                    const auto child1 = std::static_pointer_cast<Node<InType>>(node->getChild(0));
                    Eigen::Matrix<double, 2, 6> jac = -D_;
                    w_iv_inv_->backward(lhs * jac, child1, jacs);
                }
            }

            // -----------------------------------------------------------------------------
            // Factory function to create a shared instance of P2PlaneErrorGlobalPerturbEvaluator.
            // -----------------------------------------------------------------------------

            VelErrorEvaluator::Ptr velError(const Eigen::Vector2d vel_meas,
                    const Evaluable<VelErrorEvaluator::InType>::ConstPtr &w_iv_inv) {
                return VelErrorEvaluator::MakeShared(vel_meas, w_iv_inv);
            }

        }  // namespace p2p
    }  // namespace eval
}  // namespace slam