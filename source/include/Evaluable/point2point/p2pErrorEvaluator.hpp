#pragma once

#include <Eigen/Core>

#include "LGMath/LieGroupMath.hpp"
#include "Evaluable/Evaluable.hpp"
#include "Trajectory/Time.hpp"

namespace slam {
    namespace eval {
        namespace p2p {

            // -----------------------------------------------------------------------------
            /**
             * @class P2PErrorEvaluator
             * @brief Computes the point-to-point residual error for SE(3) transformations.
             *
             * This evaluator computes the error between a **reference point** and a **query point**
             * transformed by an SE(3) transformation.
             *
             * \f[
             * e = D (r - T_{rq} \cdot q)
             * \f]
             *
             * Where:
             * - \( T_{rq} \) is the transformation applied to the query point.
             * - \( r \) is the **reference point**.
             * - \( q \) is the **query point**.
             * - \( D \) extracts the translation component of the transformation.
             * - \( e \) is the computed error vector.
             */
            class P2PErrorEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
                public:
                    using Ptr = std::shared_ptr<P2PErrorEvaluator>;
                    using ConstPtr = std::shared_ptr<const P2PErrorEvaluator>;

                    using InType = slam::liemath::se3::Transformation;
                    using OutType = Eigen::Matrix<double, 3, 1>;
                    using Time = slam::traj::Time;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared pointer.
                     * @param T_rq SE(3) transformation evaluator.
                     * @param reference Reference point.
                     * @param query Query point.
                     * @return Shared pointer to the created evaluator.
                     */
                    static Ptr MakeShared(const Evaluable<InType>::ConstPtr& T_rq,
                                        const Eigen::Vector3d& reference,
                                        const Eigen::Vector3d& query);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructs the point-to-point error evaluator.
                     * @param T_rq SE(3) transformation evaluator.
                     * @param reference Reference point.
                     * @param query Query point.
                     */
                    P2PErrorEvaluator(const Evaluable<InType>::ConstPtr& T_rq,
                                    const Eigen::Vector3d& reference,
                                    const Eigen::Vector3d& query);

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
                    void getRelatedVarKeys(KeySet& keys) const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the point-to-point error between a reference and transformed query point.
                     *
                     * This function computes the error based on:
                     * 
                     * \f[
                     * e = D (p_r - T_{rq} p_q)
                     * \f]
                     *
                     * where:
                     * - \( e \) is the **point-to-point error**.
                     * - \( D \) is the **dimensional selection matrix** (typically extracting the first three components).
                     * - \( p_r \) is the **reference point**.
                     * - \( p_q \) is the **query point in the local frame**.
                     * - \( T_{rq} \) is the **rigid-body transformation**.
                     *
                     * @return The computed point-to-point error.
                     */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the forward pass of the point-to-point error and stores it in a node.
                     *
                     * This function retrieves the transformation node, applies it to the query point,
                     * and computes the error between the reference and transformed query point.
                     *
                     * @throws std::runtime_error If the transformation node is null.
                     * @return A shared pointer to the newly created node containing the error.
                     */
                    Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the Jacobian and propagates gradients in the optimization framework.
                     *
                     * This function computes the Jacobian of the error function with respect to \( T_{rq} \),
                     * using the **point-to-frame mapping function**.
                     *
                     * The Jacobian is computed as:
                     * 
                     * \f[
                     * \frac{\partial e}{\partial \xi} = -D \cdot J_{se3}
                     * \f]
                     *
                     * where \( J_{se3} \) is the **point-to-frame Jacobian** in Lie algebra.
                     *
                     * @param lhs Left-hand-side weight matrix from the optimization framework.
                     * @param node Node containing the current transformation estimate.
                     * @param jacs Container for accumulating Jacobians associated with state variables.
                     */
                    void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs, 
                                const Node<OutType>::Ptr& node, 
                                StateKeyJacobians& jacs) const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Sets the time associated with the measurement.
                     */
                    void setTime(const Time& time) noexcept {
                        time_ = time;
                        time_init_ = true;
                    }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Retrieves the timestamp for the acceleration measurement.
                     * @return Timestamp.
                     * @throws std::runtime_error if the timestamp was not initialized.
                     */
                    Time getTime() const {
                        return time_init_ ? time_ : throw std::runtime_error("[P2PErrorEvaluator::getTime] Time was not initialized");
                    }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the Jacobian of the point-to-point error with respect to the transformation.
                     *
                     * This function returns the Jacobian of the error function with respect to \( T_{rq} \),
                     * utilizing the **Lie algebra** representation.
                     *
                     * @return The computed Jacobian matrix.
                     */
                    Eigen::Matrix<double, 3, 6> getJacobianPose() const;

                private:
                    const Evaluable<InType>::ConstPtr T_rq_; ///< SE(3) transformation evaluator.
                    Eigen::Matrix<double, 3, 4> D_; ///< Selection matrix for translation extraction.
                    Eigen::Vector4d reference_; ///< Reference point in homogeneous coordinates.
                    Eigen::Vector4d query_; ///< Query point in homogeneous coordinates.

                    bool time_init_ = false; ///< Flag indicating whether time was set.
                    Time time_; ///< Measurement timestamp.
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Convenience function to create a point-to-point error evaluator.
             * @param T_rq SE(3) transformation evaluator.
             * @param reference Reference point.
             * @param query Query point.
             * @return Shared pointer to the created evaluator.
             */
            P2PErrorEvaluator::Ptr p2pError(
                const Evaluable<P2PErrorEvaluator::InType>::ConstPtr& T_rq,
                const Eigen::Vector3d& reference, const Eigen::Vector3d& query);

        }  // namespace p2p
    }  // namespace eval
}  // namespace slam
