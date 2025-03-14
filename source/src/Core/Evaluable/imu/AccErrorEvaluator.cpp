#include "Core/Evaluable/imu/AccErrorEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace imu {

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory method to create an instance.
             */
            // -----------------------------------------------------------------------------

            auto AccelerationErrorEvaluator::MakeShared(const Evaluable<PoseInType>::ConstPtr &transform,
                                                        const Evaluable<AccInType>::ConstPtr &acceleration,
                                                        const Evaluable<BiasInType>::ConstPtr &bias,
                                                        const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                                                        const ImuInType &acc_meas) -> Ptr {
                return std::make_shared<AccelerationErrorEvaluator>(transform, acceleration, bias, transform_i_to_m, acc_meas);
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructor for AccelerationErrorEvaluator.
             */
            // -----------------------------------------------------------------------------

            AccelerationErrorEvaluator::AccelerationErrorEvaluator(const Evaluable<PoseInType>::ConstPtr &transform,
                                                                const Evaluable<AccInType>::ConstPtr &acceleration,
                                                                const Evaluable<BiasInType>::ConstPtr &bias,
                                                                const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                                                                const ImuInType &acc_meas)
                : transform_(transform),
                acceleration_(acceleration),
                bias_(bias),
                transform_i_to_m_(transform_i_to_m),
                acc_meas_(acc_meas) {
                gravity_ << 0, 0, -9.8042;
                jac_accel_.setIdentity();
                jac_bias_.setIdentity();
                jac_bias_ *= -1;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Checks if the evaluator is active.
             */
            // -----------------------------------------------------------------------------

            bool AccelerationErrorEvaluator::active() const {
                return transform_->active() || acceleration_->active() || bias_->active() || transform_i_to_m_->active();
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Collects state variable keys that influence this evaluator.
             */
            // -----------------------------------------------------------------------------
            
            void AccelerationErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
                transform_->getRelatedVarKeys(keys);
                acceleration_->getRelatedVarKeys(keys);
                bias_->getRelatedVarKeys(keys);
                transform_i_to_m_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the acceleration error.
             */
            // -----------------------------------------------------------------------------

            auto AccelerationErrorEvaluator::value() const -> OutType {
                return acc_meas_ + acceleration_->value().head<3>() +
                    transform_->value().C_ba() * transform_i_to_m_->value().C_ba() * gravity_ -
                    bias_->value().head<3>();
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Forward evaluation of acceleration error.
             */
            // -----------------------------------------------------------------------------

            auto AccelerationErrorEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child1 = transform_->forward();
                const auto child2 = acceleration_->forward();
                const auto child3 = bias_->forward();
                const auto child4 = transform_i_to_m_->forward();

                const auto error = acc_meas_ + child2->value().head<3>() +
                                child1->value().C_ba() * child4->value().C_ba() * gravity_ -
                                child3->value().head<3>();

                auto node = Node<OutType>::MakeShared(error);
                node->addChild(child1);
                node->addChild(child2);
                node->addChild(child3);
                node->addChild(child4);
                return node;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes Jacobians for the acceleration error.
             */
            // -----------------------------------------------------------------------------

            void AccelerationErrorEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                                    const Node<OutType>::Ptr& node,
                                                    StateKeyJacobians& jacs) const {
                const auto child1 = std::static_pointer_cast<Node<PoseInType>>(node->getChild(0));
                const auto child2 = std::static_pointer_cast<Node<AccInType>>(node->getChild(1));
                const auto child3 = std::static_pointer_cast<Node<BiasInType>>(node->getChild(2));
                const auto child4 = std::static_pointer_cast<Node<PoseInType>>(node->getChild(3));

                const Eigen::Matrix3d C_vm = child1->value().C_ba();
                const Eigen::Matrix3d C_mi = child4->value().C_ba();

                if (transform_->active()) {
                    JacType jac = JacType::Zero();
                    jac.rightCols<3>() = -liemath::so3::hat(C_vm * C_mi * gravity_);
                    transform_->backward(lhs * jac, child1, jacs);
                }

                if (acceleration_->active()) {
                    acceleration_->backward(lhs * jac_accel_, child2, jacs);
                }

                if (bias_->active()) {
                    bias_->backward(lhs * jac_bias_, child3, jacs);
                }

                if (transform_i_to_m_->active()) {
                    JacType jac = JacType::Zero();
                    jac.rightCols<3>() = -C_vm * liemath::so3::hat(C_mi * gravity_);
                    transform_i_to_m_->backward(lhs * jac, child4, jacs);
                }
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes measurement Jacobians.
             */
            // -----------------------------------------------------------------------------

            void AccelerationErrorEvaluator::getMeasJacobians(JacType &jac_pose,
                                                            JacType &jac_accel,
                                                            JacType &jac_bias,
                                                            JacType &jac_T_mi) const {
                const Eigen::Matrix3d C_vm = transform_->value().C_ba();
                const Eigen::Matrix3d C_mi = transform_i_to_m_->value().C_ba();

                jac_pose.setZero();
                jac_pose.rightCols<3>() = -liemath::so3::hat(C_vm * C_mi * gravity_);

                jac_accel.setIdentity();
                jac_bias.setIdentity();
                jac_bias *= -1;

                jac_T_mi.setZero();
                jac_T_mi.rightCols<3>() = -C_vm * liemath::so3::hat(C_mi * gravity_);
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory function for creating an AccelerationErrorEvaluator.
             */
            // -----------------------------------------------------------------------------

            auto AccelerationError(const Evaluable<AccelerationErrorEvaluator::PoseInType>::ConstPtr &transform,
                                        const Evaluable<AccelerationErrorEvaluator::AccInType>::ConstPtr &acceleration,
                                        const Evaluable<AccelerationErrorEvaluator::BiasInType>::ConstPtr &bias,
                                        const Evaluable<AccelerationErrorEvaluator::PoseInType>::ConstPtr &transform_i_to_m,
                                        const AccelerationErrorEvaluator::ImuInType &acc_meas) -> AccelerationErrorEvaluator::Ptr {
                return std::make_shared<AccelerationErrorEvaluator>(transform, acceleration, bias, transform_i_to_m, acc_meas);
            }
        }  // namespace imu
    }  // namespace eval
}  // namespace slam
