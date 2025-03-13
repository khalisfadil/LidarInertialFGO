#pragma once

#include <Eigen/Core>
#include "Evaluable/Evaluable.hpp"

namespace slam {
    namespace eval {
        namespace p2p {

            // -----------------------------------------------------------------------------
            /**
             * @class VelErrorEvaluator
             * @brief Evaluates the velocity error for factor graph optimization.
             *
             * Computes velocity error given a velocity measurement \( v_{meas} \) and an estimated velocity state.
             * Used for velocity-based constraints in SLAM and sensor fusion.
             */
            class VelErrorEvaluator : public Evaluable<Eigen::Matrix<double, 2, 1>> {
                public:
                    using Ptr = std::shared_ptr<VelErrorEvaluator>;
                    using ConstPtr = std::shared_ptr<const VelErrorEvaluator>;
                    using InType = Eigen::Matrix<double, 6, 1>;
                    using OutType = Eigen::Matrix<double, 2, 1>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create an instance.
                     * @param vel_meas Measured velocity.
                     * @param w_iv_inv Estimated inverse velocity state.
                     * @return Shared pointer to a new evaluator instance.
                     */
                    static Ptr MakeShared(const Eigen::Vector2d& vel_meas, const Evaluable<InType>::ConstPtr &w_iv_inv);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor.
                     * @param vel_meas Measured velocity.
                     * @param w_iv_inv Estimated inverse velocity state.
                     */
                    VelErrorEvaluator(const Eigen::Vector2d& vel_meas, const Evaluable<InType>::ConstPtr &w_iv_inv);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Checks if the velocity state influences active state variables.
                     * @return True if active, otherwise false.
                     */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Collects state variable keys that influence this evaluator.
                     * @param[out] keys Set of related state keys.
                     */
                    void getRelatedVarKeys(KeySet& keys) const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the velocity error.
                     * @return Velocity error as a 2D vector.
                     */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Forward evaluation of velocity error.
                     * @return Shared pointer to the computed velocity error node.
                     */
                    Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes Jacobians for the velocity error.
                     * @param lhs Left-hand-side weight matrix.
                     * @param node Transformation estimate node.
                     * @param jacs Container for accumulating Jacobians.
                     */
                    void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                  const Node<OutType>::Ptr& node,
                                  StateKeyJacobians& jacs) const override;
                
                private:
                    
                    const Eigen::Vector2d vel_meas_;    ///< Measured velocity
                    const Evaluable<InType>::ConstPtr w_iv_inv_;    ///< Evaluated state
                    Eigen::Matrix<double, 2, 6> D_;     ///< Jacobian matrix
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory function for creating a velocity error evaluator.
             * @param vel_meas Measured velocity.
             * @param w_iv_inv Estimated inverse velocity state.
             * @return Shared pointer to the evaluator.
             */
            VelErrorEvaluator::Ptr velError(
                const Eigen::Vector2d vel_meas,
                const Evaluable<VelErrorEvaluator::InType>::ConstPtr &w_iv_inv);

        }  // namespace p2p
    }  // namespace eval
}  // namespace slam
