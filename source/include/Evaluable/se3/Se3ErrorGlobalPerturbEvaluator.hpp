#pragma once

#include <Eigen/Core>
#include "source/include/LGMath/LieGroupMath.hpp"
#include "source/include/Evaluable/Evaluable.hpp"

namespace slam {
    namespace eval {
        namespace se3 {

            // -----------------------------------------------------------------------------
            /**
             * @class SE3ErrorGlobalPerturbEvaluator
             * @brief Computes the global perturbation error between an estimated and measured SE(3) transformation.
             *
             * Given an estimated transformation \( T_{ab} \) and a measured transformation \( T_{ab}^{meas} \),
             * this evaluator computes the error in a global reference frame:
             * \f[
             * \xi = \begin{bmatrix} t_{error} \\ r_{error} \end{bmatrix}
             * \f]
             * where \( t_{error} \) is the translation error and \( r_{error} \) is the rotational error in Lie algebra.
             * This avoids SE(3) logarithm-based error formulations, making it more intuitive for optimization.
             */
            class SE3ErrorGlobalPerturbEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
                public:
                    using Ptr = std::shared_ptr<SE3ErrorGlobalPerturbEvaluator>;
                    using ConstPtr = std::shared_ptr<const SE3ErrorGlobalPerturbEvaluator>;

                    using InType = slam::liemath::se3::Transformation;
                    using OutType = Eigen::Matrix<double, 6, 1>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared pointer instance.
                     * @param T_ab Estimated SE(3) transformation \( T_{ab} \).
                     * @param T_ab_meas Measured SE(3) transformation \( T_{ab}^{meas} \).
                     * @return Shared pointer to a new SE3ErrorGlobalPerturbEvaluator instance.
                     */
                    static Ptr MakeShared(const Evaluable<InType>::ConstPtr& T_ab,
                                        const InType& T_ab_meas);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor that initializes the evaluator.
                     * @param T_ab Estimated SE(3) transformation \( T_{ab} \).
                     * @param T_ab_meas Measured SE(3) transformation \( T_{ab}^{meas} \).
                     */
                    SE3ErrorGlobalPerturbEvaluator(const Evaluable<InType>::ConstPtr& T_ab,
                                            const InType& T_ab_meas);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Checks if the transformation \( T_{ab} \) depends on active state variables.
                     * @return True if \( T_{ab} \) is active, otherwise false.
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
                     * @brief Computes the global perturbation error.
                     * @return The resulting 6D twist vector.
                     */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Forward pass: Computes and stores the SE(3) global perturbation error.
                     * @return A node containing the computed 6D twist vector.
                     */
                    Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Backward pass: Computes and accumulates Jacobians for optimization.
                     * 
                     * Given a left-hand side (LHS) weight matrix and a node from the forward pass,
                     * this method propagates gradients to the transformation \( T_{ab} \).
                     * 
                     * @param lhs Left-hand-side weight matrix.
                     * @param node Node containing the forward-pass result.
                     * @param jacs Container to store the computed Jacobians.
                     */
                    void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                const Node<OutType>::Ptr& node,
                                StateKeyJacobians& jacs) const override;

                private:
                    const Evaluable<InType>::ConstPtr T_ab_; ///< Estimated SE(3) transformation \( T_{ab} \).
                    const InType T_ab_meas_; ///< Measured SE(3) transformation \( T_{ab}^{meas} \).
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Creates an SE3ErrorGlobalPerturbEvaluator evaluator.
             *
             * This is a convenience function to create an evaluator for computing
             * \( \xi = [t_{error}, r_{error}] \).
             * 
             * @param T_ab Estimated SE(3) transformation \( T_{ab} \).
             * @param T_ab_meas Measured SE(3) transformation \( T_{ab}^{meas} \).
             * @return Shared pointer to the created evaluator.
             */
            SE3ErrorGlobalPerturbEvaluator::Ptr se3_global_perturb_error(
                const Evaluable<SE3ErrorGlobalPerturbEvaluator::InType>::ConstPtr& T_ab,
                const SE3ErrorGlobalPerturbEvaluator::InType& T_ab_meas);

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
