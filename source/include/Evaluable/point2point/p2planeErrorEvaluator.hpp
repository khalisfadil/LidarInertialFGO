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
             * @class P2PlaneErrorEvaluator
             * @brief Computes point-to-plane residual error for SLAM factor graph optimization.
             *
             * This evaluator computes the perpendicular distance from a transformed query point
             * to a reference plane. It is useful in LiDAR-based SLAM for aligning points to surfaces.
             *
             * Error Function:
             * \f[
             * e_{\text{plane}} = n^T \cdot (p_{\text{ref}} - T_{rq} \cdot p_q)
             * \f]
             * Where:
             * - \( n \) → Plane normal.
             * - \( p_{\text{ref}} \) → Reference point on the plane.
             * - \( p_q \) → Query point before transformation.
             * - \( T_{rq} \) → Transformation applied to query point.
             */
            class P2PlaneErrorEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
                public:
                    using Ptr = std::shared_ptr<P2PlaneErrorEvaluator>;
                    using ConstPtr = std::shared_ptr<const P2PlaneErrorEvaluator>;

                    using InType = slam::liemath::se3::Transformation;
                    using OutType = Eigen::Matrix<double, 1, 1>;
                    using Time = slam::traj::Time;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared instance.
                     * @param T_rq Transformation matrix.
                     * @param reference Reference point on the plane.
                     * @param query Query point to be transformed.
                     * @param normal Normal vector of the plane.
                     * @return Shared pointer to `P2PlaneErrorEvaluator`.
                     */
                    static Ptr MakeShared(const Evaluable<InType>::ConstPtr& T_rq,
                                        const Eigen::Vector3d& reference,
                                        const Eigen::Vector3d& query,
                                        const Eigen::Vector3d& normal);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor.
                     * @param T_rq Transformation matrix.
                     * @param reference Reference point on the plane.
                     * @param query Query point to be transformed.
                     * @param normal Normal vector of the plane.
                     */
                    P2PlaneErrorEvaluator(const Evaluable<InType>::ConstPtr& T_rq,
                                        const Eigen::Vector3d& reference,
                                        const Eigen::Vector3d& query,
                                        const Eigen::Vector3d& normal);

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

                    /**
                     * @brief Computes the point-to-plane error.
                     *
                     * Given a transformation \( T_{rq} \) that maps a query point to a reference frame,
                     * this function calculates the signed distance between the transformed query point
                     * and the reference point along the provided normal direction.
                     *
                     * Mathematically, the error is given by:
                     * \f[
                     * e = n^\top (p_r - (R_{rq} p_q + t_{rq}))
                     * \f]
                     * where:
                     * - \( n \) is the plane normal.
                     * - \( p_r \) is the reference point.
                     * - \( p_q \) is the query point.
                     * - \( R_{rq} \) and \( t_{rq} \) are the rotation and translation components of \( T_{rq} \).
                     *
                     * @throws std::runtime_error if `T_rq_` is null.
                     * @return The computed error as a 1×1 Eigen matrix.
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
                        return time_init_ ? time_ : throw std::runtime_error("[P2PlaneErrorEvaluator::getTime] Time was not initialized.");
                    }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the Jacobian with respect to the global pose perturbation.
                     * @return Jacobian matrix \( J \) of size (1x6).
                     */
                    Eigen::Matrix<double, 1, 6> getJacobianPose() const;

                private:
                    const Evaluable<InType>::ConstPtr T_rq_;
                    const Eigen::Vector3d reference_;
                    const Eigen::Vector3d query_;
                    const Eigen::Vector3d normal_;

                    bool time_init_ = false;
                    Time time_;
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Creates an evaluator for point-to-plane error.
             * @param T_rq Transformation matrix.
             * @param reference Reference point on the plane.
             * @param query Query point to be transformed.
             * @param normal Normal vector of the plane.
             * @return Shared pointer to the created evaluator.
             */
            P2PlaneErrorEvaluator::Ptr p2planeError(
                const Evaluable<P2PlaneErrorEvaluator::InType>::ConstPtr& T_rq,
                const Eigen::Vector3d& reference, 
                const Eigen::Vector3d& query, 
                const Eigen::Vector3d& normal);

        }  // namespace p2p
    }  // namespace eval
}  // namespace slam
