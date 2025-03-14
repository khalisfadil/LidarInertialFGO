#include "Core/Trajectory/ConstAcceleration/Evaluable/composeCurlyhatEvaluator.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {
            
            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto ComposeCurlyhatEvaluator::MakeShared(const Evaluable<InType>::ConstPtr& x,
                                          const Evaluable<InType>::ConstPtr& y)
                -> Ptr {
                return std::make_shared<ComposeCurlyhatEvaluator>(x, y);
            }

            // -----------------------------------------------------------------------------
            // ComposeCurlyhatEvaluator
            // -----------------------------------------------------------------------------

            ComposeCurlyhatEvaluator::ComposeCurlyhatEvaluator(
                const Evaluable<InType>::ConstPtr& x, const Evaluable<InType>::ConstPtr& y)
                : x_(x), y_(y) {}

            // -----------------------------------------------------------------------------
            // active
            // -----------------------------------------------------------------------------

            bool ComposeCurlyhatEvaluator::active() const {
                return x_->active() || y_->active();
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void ComposeCurlyhatEvaluator::getRelatedVarKeys(KeySet& keys) const {
                x_->getRelatedVarKeys(keys);
                y_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // value
            // -----------------------------------------------------------------------------

            auto ComposeCurlyhatEvaluator::value() const -> OutType {
                return liemath::se3::curlyhat(x_->value()) * y_->value();
            }

            // -----------------------------------------------------------------------------
            // forward
            // -----------------------------------------------------------------------------

            auto ComposeCurlyhatEvaluator::forward() const -> eval::Node<OutType>::Ptr {
                const auto x = x_->forward(), y = y_->forward();
                const auto node = eval::Node<OutType>::MakeShared(liemath::se3::curlyhat(x->value()) * y->value());

                for (const auto& child : {x, y}) node->addChild(child);
                return node;
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            void ComposeCurlyhatEvaluator::backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs, 
                                                    const slam::eval::Node<OutType>::Ptr& node,
                                                    slam::eval::StateKeyJacobians& jacs) const {
                const auto x = std::static_pointer_cast<eval::Node<InType>>(node->getChild(0));
                const auto y = std::static_pointer_cast<eval::Node<InType>>(node->getChild(1));

                if (x_->active()) 
                    x_->backward(-lhs * liemath::se3::curlyhat(y->value()), x, jacs);

                if (y_->active()) 
                    y_->backward(lhs * liemath::se3::curlyhat(x->value()), y, jacs);
            }

            // -----------------------------------------------------------------------------
            // backward
            // -----------------------------------------------------------------------------

            auto compose_curlyhat(const ComposeCurlyhatEvaluator::ConstPtr& x,
                                const ComposeCurlyhatEvaluator::ConstPtr& y) -> ComposeCurlyhatEvaluator::Ptr {
                return ComposeCurlyhatEvaluator::MakeShared(x, y);
            }

        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
