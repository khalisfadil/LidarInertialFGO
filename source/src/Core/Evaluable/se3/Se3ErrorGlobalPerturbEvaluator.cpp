#include "Core/Evaluable/se3/Se3ErrorGlobalPerturbEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace se3 {

        // ----------------------------------------------------------------------------
        // MakeShared
        // ----------------------------------------------------------------------------

        auto SE3ErrorGlobalPerturbEvaluator::MakeShared(const Evaluable<InType>::ConstPtr& T_ab,
                                                const InType& T_ab_meas) -> Ptr {
            return std::make_shared<SE3ErrorGlobalPerturbEvaluator>(T_ab, T_ab_meas);
        }

        // ----------------------------------------------------------------------------
        // SE3ErrorGlobalPerturbEvaluator
        // ----------------------------------------------------------------------------

        SE3ErrorGlobalPerturbEvaluator::SE3ErrorGlobalPerturbEvaluator(const Evaluable<InType>::ConstPtr& T_ab,
                                                            const InType& T_ab_meas)
            : T_ab_(T_ab), T_ab_meas_(T_ab_meas) {}

        // ----------------------------------------------------------------------------
        // active
        // ----------------------------------------------------------------------------

        bool SE3ErrorGlobalPerturbEvaluator::active() const {
            return T_ab_->active();
        }

        // ----------------------------------------------------------------------------
        // getRelatedVarKeys
        // ----------------------------------------------------------------------------

        void SE3ErrorGlobalPerturbEvaluator::getRelatedVarKeys(KeySet& keys) const {
            T_ab_->getRelatedVarKeys(keys);
        }

        // ----------------------------------------------------------------------------
        // value
        // ----------------------------------------------------------------------------

        auto SE3ErrorGlobalPerturbEvaluator::value() const -> OutType {
            Eigen::Matrix<double, 6, 1> out = Eigen::Matrix<double, 6, 1>::Zero();
            const Eigen::Matrix4d T_ab = T_ab_->evaluate().matrix();
            const Eigen::Matrix4d T_ab_meas = T_ab_meas_.matrix();
            
            out.block<3, 1>(0, 0) = T_ab.block<3, 1>(0, 3) - T_ab_meas.block<3, 1>(0, 3);
            out.block<3, 1>(3, 0) = slam::liemath::so3::rot2vec(T_ab_meas.block<3, 3>(0, 0).transpose() * T_ab.block<3, 3>(0, 0));
            
            return out;
        }

        // ----------------------------------------------------------------------------
        // forward
        // ----------------------------------------------------------------------------

        auto SE3ErrorGlobalPerturbEvaluator::forward() const -> Node<OutType>::Ptr {
            const auto child = T_ab_->forward();
            const Eigen::Matrix4d T_ab = child->value().matrix();
            const Eigen::Matrix4d T_ab_meas = T_ab_meas_.matrix();
            
            Eigen::Matrix<double, 6, 1> value = Eigen::Matrix<double, 6, 1>::Zero();
            value.block<3, 1>(0, 0) = T_ab.block<3, 1>(0, 3) - T_ab_meas.block<3, 1>(0, 3);
            value.block<3, 1>(3, 0) = slam::liemath::so3::rot2vec(T_ab_meas.block<3, 3>(0, 0).transpose() * T_ab.block<3, 3>(0, 0));

            auto node = Node<OutType>::MakeShared(value);
            node->addChild(child);
            return node;
        }

        // ----------------------------------------------------------------------------
        // backward
        // ----------------------------------------------------------------------------

        void SE3ErrorGlobalPerturbEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                                const Node<OutType>::Ptr& node,
                                                StateKeyJacobians& jacs) const {
            if (!node || node->size() < 1) {
                throw std::runtime_error("[SE3ErrorGlobalPerturbEvaluator::backward] Node has insufficient children.");
            }

            auto child_base = node->getChild(0);
            if (!child_base) {
                throw std::runtime_error("[SE3ErrorGlobalPerturbEvaluator::backward] Null child node encountered.");
            }

            auto child = std::static_pointer_cast<Node<InType>>(child_base);
            if (!child || !child->hasValue()) {
                throw std::runtime_error("[SE3ErrorGlobalPerturbEvaluator::backward] Invalid child node.");
            }

            if (T_ab_->active()) {
                Eigen::Vector3d phi = node->value().block<3, 1>(3, 0);
                Eigen::Matrix<double, 6, 6> jac = Eigen::Matrix<double, 6, 6>::Zero();
                const Eigen::Matrix4d T_ab = child->value().matrix();

                // Compute Jacobian components
                jac.block<3, 3>(0, 0) = T_ab.block<3, 3>(0, 0);  // Rotation matrix part
                jac.block<3, 3>(3, 3) = slam::liemath::so3::vec2jacinv(-phi);  // SO(3) inverse Jacobian

                // Propagate gradients
                T_ab_->backward(lhs * jac, child, jacs);
            }
        }

        // ----------------------------------------------------------------------------
        // se3_global_perturb_error
        // ----------------------------------------------------------------------------

        SE3ErrorGlobalPerturbEvaluator::Ptr se3_global_perturb_error(
            const Evaluable<SE3ErrorGlobalPerturbEvaluator::InType>::ConstPtr& T_ab,
            const SE3ErrorGlobalPerturbEvaluator::InType& T_ab_meas) {
            return SE3ErrorGlobalPerturbEvaluator::MakeShared(T_ab, T_ab_meas);
        }

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
