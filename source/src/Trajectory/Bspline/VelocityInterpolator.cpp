#include "source/include/Trajectory/Bspline/VelocityInterpolator.hpp"

#include "source/include/Evaluable/vspace/Evaluables.hpp"

namespace slam {
    namespace traj {
        namespace bspline {

            // -----------------------------------------------------------------------------
            // B
            // -----------------------------------------------------------------------------

            const Eigen::Matrix4d VelocityInterpolator::B = (Eigen::Matrix4d() <<
                1.,  4.,  1.,  0.,
                -3.,  0.,  3.,  0.,
                3., -6.,  3.,  0.,
                -1.,  3., -3.,  1.
            ).finished() / 6.0;

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            VelocityInterpolator::Ptr VelocityInterpolator::MakeShared(
                const slam::traj::Time& time,
                const Variable::ConstPtr& k1,
                const Variable::ConstPtr& k2,
                const Variable::ConstPtr& k3,
                const Variable::ConstPtr& k4) {
                return std::make_shared<VelocityInterpolator>(time, k1, k2, k3, k4);
            }

            // -----------------------------------------------------------------------------
            // VelocityInterpolator
            // -----------------------------------------------------------------------------

            VelocityInterpolator::VelocityInterpolator(
                const slam::traj::Time& time,
                const Variable::ConstPtr& k1,
                const Variable::ConstPtr& k2,
                const Variable::ConstPtr& k3,
                const Variable::ConstPtr& k4)
                : k1_(k1), k2_(k2), k3_(k3), k4_(k4) {

                // Compute time-dependent interpolation weights
                double tau = (time - k2_->getTime()).seconds();
                double T = (k3_->getTime() - k2_->getTime()).seconds();
                double ratio = tau / T;
                double ratio2 = ratio * ratio;
                double ratio3 = ratio2 * ratio;

                Eigen::Vector4d u;
                u << 1., ratio, ratio2, ratio3;

                w_ = (u.transpose() * B).transpose();
            }

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------

            bool VelocityInterpolator::active() const {
                return k1_->getC()->active() || k2_->getC()->active() ||
                       k3_->getC()->active() || k4_->getC()->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void VelocityInterpolator::getRelatedVarKeys(KeySet& keys) const {
                k1_->getC()->getRelatedVarKeys(keys);
                k2_->getC()->getRelatedVarKeys(keys);
                k3_->getC()->getRelatedVarKeys(keys);
                k4_->getC()->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto VelocityInterpolator::value() const -> OutType {
                return w_(0) * k1_->getC()->value() +
                       w_(1) * k2_->getC()->value() +
                       w_(2) * k3_->getC()->value() +
                       w_(3) * k4_->getC()->value();
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto VelocityInterpolator::forward() const -> slam::eval::Node<OutType>::Ptr {
                const auto k1 = k1_->getC()->forward();
                const auto k2 = k2_->getC()->forward();
                const auto k3 = k3_->getC()->forward();
                const auto k4 = k4_->getC()->forward();

                // Compute interpolated velocity
                const auto value = w_(0) * k1->value() +
                                   w_(1) * k2->value() +
                                   w_(2) * k3->value() +
                                   w_(3) * k4->value();

                auto node = slam::eval::Node<OutType>::MakeShared(value);
                node->addChild(k1);
                node->addChild(k2);
                node->addChild(k3);
                node->addChild(k4);

                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void VelocityInterpolator::backward(
                const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                const slam::eval::Node<OutType>::Ptr& node,
                slam::eval::StateKeyJacobians& jacs) const {

                for (size_t i = 0; i < 4; ++i) {
                    auto& control_point = (i == 0) ? k1_ : (i == 1) ? k2_ : (i == 2) ? k3_ : k4_;
                    if (control_point->getC()->active()) {
                        auto child = std::dynamic_pointer_cast<slam::eval::Node<CType>>(node->getChild(i));
                        if (child) {
                            control_point->getC()->backward(lhs * w_(i), child, jacs);
                        }
                    }
                }
            }

        }  // namespace bspline
    }  // namespace traj
}  // namespace slam
