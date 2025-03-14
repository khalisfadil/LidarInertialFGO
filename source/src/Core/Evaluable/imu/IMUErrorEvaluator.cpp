#include "Core/Evaluable/imu/IMUErrorEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace imu {
            
            // -----------------------------------------------------------------------------
            // @brief Factory method to create an instance.
            // -----------------------------------------------------------------------------

            auto IMUErrorEvaluator::MakeShared(const Evaluable<PoseInType>::ConstPtr &transform,
                                        const Evaluable<VelInType>::ConstPtr &velocity,
                                        const Evaluable<AccInType>::ConstPtr &acceleration,
                                        const Evaluable<BiasInType>::ConstPtr &bias,
                                        const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                                        const ImuInType &imu_meas) -> Ptr {
                return std::make_shared<IMUErrorEvaluator>(transform, velocity, acceleration,
                                             bias, transform_i_to_m, imu_meas);
            }

            // -----------------------------------------------------------------------------
            // @brief Constructor for GyroErrorEvaluatorSE2.
            // -----------------------------------------------------------------------------

            IMUErrorEvaluator::IMUErrorEvaluator(const Evaluable<PoseInType>::ConstPtr &transform,
                                                    const Evaluable<VelInType>::ConstPtr &velocity,
                                                    const Evaluable<AccInType>::ConstPtr &acceleration,
                                                    const Evaluable<BiasInType>::ConstPtr &bias,
                                                    const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                                                    const ImuInType &imu_meas)
                                                    : transform_(transform),
                                                        velocity_(velocity),
                                                        acceleration_(acceleration),
                                                        bias_(bias),
                                                        transform_i_to_m_(transform_i_to_m),
                                                        imu_meas_(imu_meas) {
                gravity_(2, 0) = -9.8042;
            }


            // -----------------------------------------------------------------------------
            // @brief Checks if the evaluator is active.
            // -----------------------------------------------------------------------------
            
            bool IMUErrorEvaluator::active() const {
                return transform_->active() || velocity_->active() ||
                        acceleration_->active() || bias_->active() ||
                        transform_i_to_m_->active();
            }

            // -----------------------------------------------------------------------------
            // @brief Collects state variable keys that influence this evaluator.
            // -----------------------------------------------------------------------------
            
            void IMUErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
                transform_->getRelatedVarKeys(keys);
                velocity_->getRelatedVarKeys(keys);
                acceleration_->getRelatedVarKeys(keys);
                bias_->getRelatedVarKeys(keys);
                transform_i_to_m_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // @brief Computes the gyroscope error.
            // -----------------------------------------------------------------------------

            auto IMUErrorEvaluator::value() const -> OutType {
                const Eigen::Matrix3d C_vm = transform_->value().C_ba();
                const Eigen::Matrix3d C_mi = transform_i_to_m_->value().C_ba();
                
                OutType error = OutType::Zero();
                error.head<3>() = imu_meas_.head<3>() + acceleration_->value().head<3>() + C_vm * C_mi * gravity_ - bias_->value().head<3>();
                error.tail<3>() = imu_meas_.tail<3>() + velocity_->value().tail<3>() - bias_->value().tail<3>();
                
                return error;
            }

            // -----------------------------------------------------------------------------
            // @brief Forward evaluation of gyroscope error.
            // -----------------------------------------------------------------------------

            auto IMUErrorEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto child1 = transform_->forward();
                const auto child2 = velocity_->forward();
                const auto child3 = acceleration_->forward();
                const auto child4 = bias_->forward();
                const auto child5 = transform_i_to_m_->forward();

                const Eigen::Matrix3d C_vm = child1->value().C_ba();
                const Eigen::Matrix3d C_mi = child5->value().C_ba();
                
                OutType error = OutType::Zero();
                error.head<3>() = imu_meas_.head<3>() + child3->value().head<3>() + C_vm * C_mi * gravity_ - child4->value().head<3>();
                error.tail<3>() = imu_meas_.tail<3>() + child2->value().tail<3>() - child4->value().tail<3>();

                auto node = Node<OutType>::MakeShared(error);
                node->addChild(child1);
                node->addChild(child2);
                node->addChild(child3);
                node->addChild(child4);
                node->addChild(child5);

                return node;
            }

            // -----------------------------------------------------------------------------
            // @brief Computes Jacobians for the gyroscope error.
            // -----------------------------------------------------------------------------

            void IMUErrorEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                                    const Node<OutType>::Ptr& node,
                                                    StateKeyJacobians& jacs) const {
                const auto child1 = std::static_pointer_cast<Node<PoseInType>>(node->getChild(0));
                const auto child2 = std::static_pointer_cast<Node<VelInType>>(node->getChild(1));
                const auto child3 = std::static_pointer_cast<Node<AccInType>>(node->getChild(2));
                const auto child4 = std::static_pointer_cast<Node<BiasInType>>(node->getChild(3));
                const auto child5 = std::static_pointer_cast<Node<PoseInType>>(node->getChild(4));

                if (velocity_->active()) {
                    Eigen::Matrix<double, 6, 6> jac = Eigen::Matrix<double, 6, 6>::Zero();
                    jac.bottomRightCorner<3, 3>().setIdentity();
                    velocity_->backward(lhs * jac, child2, jacs);
                }

                if (acceleration_->active()) {
                    Eigen::Matrix<double, 6, 6> jac = Eigen::Matrix<double, 6, 6>::Zero();
                    jac.topLeftCorner<3, 3>().setIdentity();
                    acceleration_->backward(lhs * jac, child3, jacs);
                }

                if (bias_->active()) {
                    bias_->backward(lhs * Eigen::Matrix<double, 6, 6>::Identity() * -1, child4, jacs);
                }

                if (transform_i_to_m_->active()) {
                    Eigen::Matrix<double, 6, 6> jac = Eigen::Matrix<double, 6, 6>::Zero();
                    jac.topRightCorner<3, 3>() = -child1->value().C_ba() * liemath::so3::hat(child5->value().C_ba() * gravity_);
                    transform_i_to_m_->backward(lhs * jac, child5, jacs);
                }
            }

            // -----------------------------------------------------------------------------
            // @brief Factory function for creating a GyroErrorEvaluatorSE2.
            // -----------------------------------------------------------------------------

            IMUErrorEvaluator::Ptr imuError(const Evaluable<IMUErrorEvaluator::PoseInType>::ConstPtr &transform,
                                                const Evaluable<IMUErrorEvaluator::VelInType>::ConstPtr &velocity,
                                                const Evaluable<IMUErrorEvaluator::AccInType>::ConstPtr &acceleration,
                                                const Evaluable<IMUErrorEvaluator::BiasInType>::ConstPtr &bias,
                                                const Evaluable<IMUErrorEvaluator::PoseInType>::ConstPtr &transform_i_to_m,
                                                const IMUErrorEvaluator::ImuInType &imu_meas) {
                return IMUErrorEvaluator::MakeShared(transform, velocity, acceleration, bias,
                                                transform_i_to_m, imu_meas);
            }
        }  // namespace imu
    }  // namespace eval
}  // namespace slam
