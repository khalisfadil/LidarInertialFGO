#include "source/include/Evaluable/vspace/MatrixMultEvaluator.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // ----------------------------------------------------------------------------
            // MakeShared
            // ----------------------------------------------------------------------------

            template <int ROW, int COL>
            auto MatrixMultEvaluator<ROW, COL>::MakeShared(
                const typename Evaluable<InType>::ConstPtr& v, 
                const Eigen::Ref<const MatType>& s) -> Ptr {
                return std::make_shared<MatrixMultEvaluator>(v, s);
            }

            // ----------------------------------------------------------------------------
            // MatrixMultEvaluator
            // ----------------------------------------------------------------------------

            template <int ROW, int COL>
            MatrixMultEvaluator<ROW, COL>::MatrixMultEvaluator(
                const typename Evaluable<InType>::ConstPtr& v, 
                const Eigen::Ref<const MatType>& s)
                : v_(v), s_(s) {}

            // ----------------------------------------------------------------------------
            // active
            // ----------------------------------------------------------------------------

            template <int ROW, int COL>
            bool MatrixMultEvaluator<ROW, COL>::active() const {
                return v_->active();
            }

            // ----------------------------------------------------------------------------
            // getRelatedVarKeys
            // ----------------------------------------------------------------------------

            template <int ROW, int COL>
            void MatrixMultEvaluator<ROW, COL>::getRelatedVarKeys(typename Evaluable<OutType>::KeySet& keys) const {
                v_->getRelatedVarKeys(keys);
            }

            // ----------------------------------------------------------------------------
            // value
            // ----------------------------------------------------------------------------

            template <int ROW, int COL>
            auto MatrixMultEvaluator<ROW, COL>::value() const -> OutType {
                OutType result;
                result.noalias() = s_ * v_->value();
                return result;
            }

            // ----------------------------------------------------------------------------
            // forward
            // ----------------------------------------------------------------------------

            template <int ROW, int COL>
            auto MatrixMultEvaluator<ROW, COL>::forward() const -> typename Node<OutType>::Ptr {
                const auto child = v_->forward();
                OutType value;
                value.noalias() = s_ * child->value();

                const auto node = Node<OutType>::MakeShared(Eigen::Ref<const OutType>(value));
                node->addChild(child);
                return node;
            }

            // ----------------------------------------------------------------------------
            // backward
            // ----------------------------------------------------------------------------

            template <int ROW, int COL>
            void MatrixMultEvaluator<ROW, COL>::backward(
                const Eigen::MatrixXd& lhs, 
                const typename Node<OutType>::Ptr& node,
                StateKeyJacobians& jacs) const {
                
                const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
                if (v_->active()) {
                    v_->backward(lhs * s_.eval(), child, jacs);
                }
            }

            // ----------------------------------------------------------------------------
            // mmult
            // ----------------------------------------------------------------------------

            template <int ROW, int COL>
            typename MatrixMultEvaluator<ROW, COL>::Ptr mmult(
                const typename Evaluable<typename MatrixMultEvaluator<ROW, COL>::InType>::ConstPtr& v,
                const typename MatrixMultEvaluator<ROW, COL>::MatType& s) {
                return MatrixMultEvaluator<ROW, COL>::MakeShared(v, s);
            }

            // Explicit template instantiations
            template class MatrixMultEvaluator<1, 1>;
            template class MatrixMultEvaluator<2, 2>;
            template class MatrixMultEvaluator<3, 3>;

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam
