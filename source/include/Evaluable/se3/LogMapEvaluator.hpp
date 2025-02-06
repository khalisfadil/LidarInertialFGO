#pragma once

#include <Eigen/Core>
#include "source/include/LGMath/LieGroupMath.hpp"
#include "source/include/Evaluable/Evaluable.hpp"

namespace slam {
    namespace eval {
        namespace se3 {

            // -----------------------------------------------------------------------------
            /**
             * @class LogMapEvaluator
             * @brief Computes the logarithmic map from an SE(3) transformation to a 6D twist vector.
             *
             * Given a transformation \( T \), this evaluator computes:
             * \f[
             * \xi = \log(T)
             * \f]
             * where \( \xi \) is the Lie algebra representation (6D twist vector).
             * This is crucial for parameterizing SE(3) transformations in optimization problems such as SLAM.
             */
            class LogMapEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
                public:
                    using Ptr = std::shared_ptr<LogMapEvaluator>;
                    using ConstPtr = std::shared_ptr<const LogMapEvaluator>;

                    using InType = slam::liemath::se3::Transformation;
                    using OutType = Eigen::Matrix<double, 6, 1>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared pointer instance.
                     * @param transform The SE(3) transformation \( T \).
                     * @return Shared pointer to a new LogMapEvaluator instance.
                     */
                    static Ptr MakeShared(const Evaluable<InType>::ConstPtr& transform);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor that initializes the evaluator.
                     * @param transform The SE(3) transformation \( T \).
                     */
                    LogMapEvaluator(const Evaluable<InType>::ConstPtr& transform);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Checks if the transformation \( T \) depends on active state variables.
                     * @return True if \( T \) is active, otherwise false.
                     */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Collects the state variable keys influencing this evaluator.
                     * @param[out] keys The set of state keys related to this evaluator.
                     */
                    void getRelatedVarKeys(KeySet &keys) const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the logarithmic map \( \xi = \log(T) \).
                     * @return The resulting 6D twist vector.
                     */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Forward pass: Computes and stores the twist vector representation of \( T \).
                     * @return A node containing the computed 6D twist vector.
                     */
                    Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Backward pass: Computes and accumulates Jacobians for optimization.
                     * 
                     * Given a left-hand side (LHS) weight matrix and a node from the forward pass,
                     * this method propagates gradients to the transformation \( T \).
                     * 
                     * @param lhs Left-hand-side weight matrix.
                     * @param node Node containing the forward-pass result.
                     * @param jacs Container to store the computed Jacobians.
                     */
                    void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                const Node<OutType>::Ptr& node,
                                StateKeyJacobians& jacs) const override;

                private:
                    const Evaluable<InType>::ConstPtr transform_; ///< SE(3) transformation \( T \).
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Creates a LogMapEvaluator evaluator.
             *
             * This is a convenience function to create an evaluator for computing \( \xi = \log(T) \).
             * 
             * @param transform The SE(3) transformation \( T \).
             * @return Shared pointer to the created evaluator.
             */
            LogMapEvaluator::Ptr tran2vec(const Evaluable<LogMapEvaluator::InType>::ConstPtr& transform);

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
