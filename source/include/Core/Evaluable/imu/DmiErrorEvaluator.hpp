#pragma once

#include <Eigen/Core>
#include "LGMath/LieGroupMath.hpp"
#include "Core/Evaluable/Evaluable.hpp"
#include "Core/Trajectory/Time.hpp"

namespace slam {
    namespace eval {
        namespace imu {

            // -----------------------------------------------------------------------------
            /**
             * @class DMIErrorEvaluator
             * @brief Evaluates Distance Measurement Instrument (DMI) error for factor graph optimization.
             *
             * Computes the DMI error given a measured distance and an estimated velocity state.
             * Used in wheel odometry constraints for SLAM and sensor fusion.
             */
            class DMIErrorEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
                public:
                    using Ptr = std::shared_ptr<DMIErrorEvaluator>;
                    using ConstPtr = std::shared_ptr<const DMIErrorEvaluator>;
                    using VelInType = Eigen::Matrix<double, 6, 1>;
                    using DMIInType = double;
                    using ScaleInType = Eigen::Matrix<double, 1, 1>;
                    using OutType = Eigen::Matrix<double, 1, 1>;
                    using Time = slam::traj::Time;
                    using JacType = Eigen::Matrix<double, 1, 6>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create an instance.
                     * @param velocity Estimated velocity state.
                     * @param scale Scale factor for DMI correction.
                     * @param dmi_meas Measured DMI distance.
                     * @return Shared pointer to a new evaluator instance.
                     */
                    static Ptr MakeShared(const Evaluable<VelInType>::ConstPtr &velocity,
                                        const Evaluable<ScaleInType>::ConstPtr &scale,
                                        DMIInType dmi_meas);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor.
                     * @param velocity Estimated velocity state.
                     * @param scale Scale factor for DMI correction.
                     * @param dmi_meas Measured DMI distance.
                     */
                    DMIErrorEvaluator(const Evaluable<VelInType>::ConstPtr &velocity,
                                    const Evaluable<ScaleInType>::ConstPtr &scale,
                                    DMIInType dmi_meas);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Checks if the DMI error is influenced by active state variables.
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
                     * @brief Computes the DMI error.
                     * @return DMI error as a 1x1 matrix.
                     */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Forward evaluation of DMI error.
                     * @return Shared pointer to the computed DMI error node.
                     */
                    Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes Jacobians for the DMI error.
                     * @param lhs Left-hand-side weight matrix.
                     * @param node Velocity state node.
                     * @param jacs Container for accumulating Jacobians.
                     */
                    void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                                const Node<OutType>::Ptr& node,
                                StateKeyJacobians& jacs) const override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Sets the timestamp for the DMI measurement.
                     * @param time Timestamp.
                     */
                    void setTime(Time time) { 
                        time_ = time; 
                        time_init_ = true; 
                    }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Retrieves the timestamp for the DMI measurement.
                     * @return Timestamp.
                     * @throws std::runtime_error if the timestamp was not initialized.
                     */
                    Time getTime() const {
                        return time_init_ ? time_ : throw std::runtime_error("[DMIErrorEvaluator::getTime] Time was not initialized");
                    }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Computes measurement Jacobians.
                     * @param[out] jac_vel Jacobian w.r.t. velocity.
                     * @param[out] jac_scale Jacobian w.r.t. scale.
                     */
                    void getMeasJacobians(JacType &jac_vel, Eigen::Matrix<double, 1, 1> &jac_scale) const;

                private:
                    const Evaluable<VelInType>::ConstPtr velocity_;  ///< Estimated velocity state.
                    const Evaluable<ScaleInType>::ConstPtr scale_;  ///< Scale factor for DMI.
                    const DMIInType dmi_meas_;                      ///< Measured DMI distance.

                    JacType jac_vel_ = JacType::Zero();             ///< Velocity Jacobian.
                    Eigen::Matrix<double, 1, 1> jac_scale_ = Eigen::Matrix<double, 1, 1>::Zero(); ///< Scale Jacobian.

                    Time time_;                                     ///< Timestamp of measurement.
                    bool time_init_ = false;                        ///< Flag indicating if time was set.
            };

            // -----------------------------------------------------------------------------
            /**
             * @brief Factory function for creating a DMIErrorEvaluator.
             * @param velocity Estimated velocity state.
             * @param scale Scale factor for DMI correction.
             * @param dmi_meas Measured DMI distance.
             * @return Shared pointer to the evaluator.
             */
            auto DMIError(const Evaluable<DMIErrorEvaluator::VelInType>::ConstPtr &velocity,
                                            const Evaluable<DMIErrorEvaluator::ScaleInType>::ConstPtr &scale,
                                            DMIErrorEvaluator::DMIInType dmi_meas) -> DMIErrorEvaluator::Ptr;

        }  // namespace imu
    }  // namespace eval
}  // namespace slam
