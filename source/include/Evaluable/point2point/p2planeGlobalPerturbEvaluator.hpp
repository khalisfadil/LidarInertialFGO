#pragma once

#include <Eigen/Core>

#include "source/include/LGMath/LieGroupMath.hpp"
#include "source/include/Evaluable/Evaluable.hpp"
#include "source/include/Trajectory/Time.hpp"

namespace slam {
    namespace eval {
        namespace p2p {

            // -----------------------------------------------------------------------------
            /**
             * @class P2PlaneErrorGlobalPerturbEvaluator
             * @brief Computes point-to-plane error with Jacobians using right-hand-side global perturbations.
             *
             * This class is designed to be used with `SE3StateVarGlobalPerturb`, ensuring consistency in factor graphs.
             * It enforces the constraint:
             * \f[
             * e = n^T \cdot \left( r - (C * q + t) \right)
             * \f]
             * where \( C \) is the rotation, \( t \) is the translation, \( q \) is the query point,
             * and \( n \) is the normal vector of the plane.
             */
            class P2PlaneErrorGlobalPerturbEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
                public:
                    using Ptr = std::shared_ptr<P2PlaneErrorGlobalPerturbEvaluator>;
                    using ConstPtr = std::shared_ptr<const P2PlaneErrorGlobalPerturbEvaluator>;

                    using InType = slam::liemath::se3::Transformation;
                    using OutType = Eigen::Matrix<double, 1, 1>;
                    using Time = slam::traj::Time;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create an instance.
                     * @param T_rq The SE(3) transformation between reference and query frames.
                     * @param reference Reference point on the plane.
                     * @param query Query point.
                     * @param normal Plane normal vector.
                     * @return Shared pointer to a new evaluator instance.
                     */
                    static Ptr MakeShared(const Evaluable<InType>::ConstPtr& T_rq,
                                        const Eigen::Vector3d& reference,
                                        const Eigen::Vector3d& query,
                                        const Eigen::Vector3d& normal);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor.
                     * @param T_rq The SE(3) transformation between reference and query frames.
                     * @param reference Reference point on the plane.
                     * @param query Query point.
                     * @param normal Plane normal vector.
                     */
                    P2PlaneErrorGlobalPerturbEvaluator(const Evaluable<InType>::ConstPtr& T_rq,
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

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the point-to-plane error for a transformed query point.
                     *
                     * Given a transformation \( T_{rq} \) that maps a query point \( p_q \) into the reference frame,
                     * this function computes the signed **point-to-plane error**, defined as:
                     * \f[
                     * e = n^\top (p_r - (R_{rq} p_q + t_{rq}))
                     * \f]
                     * where:
                     * - \( n \) is the **unit normal** of the reference plane.
                     * - \( p_r \) is the **reference point** on the plane.
                     * - \( p_q \) is the **query point** in the local frame.
                     * - \( R_{rq}, t_{rq} \) are the **rotation matrix** and **translation vector** of \( T_{rq} \).
                     *
                     * This error metric is commonly used in **ICP-based plane constraints** for scan registration.
                     *
                     * @throws std::runtime_error if `T_rq_` is null.
                     * @return The computed error as an Eigen::Matrix<1,1> scalar.
                     */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the forward pass of the point-to-plane error with right-hand-side global perturbations.
                     *
                     * This function evaluates the **point-to-plane error** based on the transformed query point,
                     * using the current transformation estimate \( T_{rq} \).
                     * The error is defined as:
                     * \f[
                     * e = n^\top (p_r - (R_{rq} p_q + t_{rq}))
                     * \f]
                     * where:
                     * - \( n \) is the **unit normal** of the reference plane.
                     * - \( p_r \) is the **reference point** on the plane.
                     * - \( p_q \) is the **query point** in the local frame.
                     * - \( R_{rq}, t_{rq} \) are the **rotation matrix** and **translation vector** of \( T_{rq} \).
                     *
                     * This function propagates the computed error to a node that tracks its value and dependencies.
                     *
                     * @throws std::runtime_error if the transformation node is null.
                     * @return A shared pointer to the computed node containing the point-to-plane error.
                     */
                    Node<OutType>::Ptr forward() const override;
                    
                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes and propagates Jacobians for the point-to-plane error with global perturbations.
                     *
                     * This function computes the Jacobian of the **point-to-plane error** with respect to the 
                     * right-hand-side **global perturbation** of the transformation \( T_{rq} \), where:
                     * 
                     * \f[
                     * e = n^\top (p_r - (R_{rq} p_q + t_{rq}))
                     * \f]
                     *
                     * The Jacobian is computed as:
                     * 
                     * \f[
                     * \frac{\partial e}{\partial \delta_r} = - n^\top R_{rq}
                     * \f]
                     * 
                     * \f[
                     * \frac{\partial e}{\partial \delta_\phi} = n^\top R_{rq} [p_q]_\times
                     * \f]
                     *
                     * where:
                     * - \( n \) is the **unit normal** of the reference plane.
                     * - \( p_r \) is the **reference point** on the plane.
                     * - \( p_q \) is the **query point** in the local frame.
                     * - \( R_{rq}, t_{rq} \) are the **rotation matrix** and **translation vector** of \( T_{rq} \).
                     * - \( [p_q]_\times \) is the **skew-symmetric matrix** representing the cross product.
                     * 
                     * The function retrieves the transformation node, validates its state, 
                     * and propagates the computed Jacobian to the optimization framework.
                     *
                     * @throws std::runtime_error If the transformation node is invalid.
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
                     * @brief Retrieves the measurement time.
                     * @throws std::runtime_error if the time has not been initialized.
                     */
                    Time getTime() const {
                        if (!time_init_) {
                            throw std::runtime_error("[P2PlaneErrorGlobalPerturbEvaluator::getTime] Time was not initialized.");
                        }
                        return time_;
                    }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes the Jacobian of the point-to-plane error with respect to SE(3) pose perturbation.
                     *
                     * Given a transformation \( T_{rq} \) that maps a query point \( p_q \) into the reference frame,
                     * this function computes the Jacobian of the point-to-plane error with respect to perturbations in SE(3).
                     *
                     * In a **global perturbation model**, we apply small perturbations on the right-hand side:
                     * \f[
                     * C \leftarrow C \cdot \exp(\delta\phi^\wedge), \quad r \leftarrow r + \delta r
                     * \f]
                     * where \( \delta x = [\delta r; \delta \phi] \) represents translational and rotational perturbations.
                     *
                     * The error function is defined as:
                     * \f[
                     * e = n^\top (p_r - (R_{rq} p_q + t_{rq}))
                     * \f]
                     * Its Jacobian is:
                     * \f[
                     * J = \left[ -n^\top, n^\top R_{rq} \cdot \hat{p_q} \right]
                     * \f]
                     *
                     * @throws std::runtime_error if `T_rq_` is null.
                     * @return The computed Jacobian as a 1×6 Eigen matrix.
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
             * @brief Factory function for creating a P2PlaneErrorGlobalPerturbEvaluator.
             * @param T_rq The SE(3) transformation between reference and query frames.
             * @param reference Reference point on the plane.
             * @param query Query point.
             * @param normal Plane normal vector.
             * @return Shared pointer to the created evaluator.
             */
            P2PlaneErrorGlobalPerturbEvaluator::Ptr p2planeGlobalError(
                const Evaluable<P2PlaneErrorGlobalPerturbEvaluator::InType>::ConstPtr& T_rq,
                const Eigen::Vector3d& reference,
                const Eigen::Vector3d& query,
                const Eigen::Vector3d& normal);

        }  // namespace p2p
    }  // namespace eval
}  // namespace slam
