#pragma once

#include <Eigen/Core>

#include "source/include/Trajectory/ConstAcceleration/Variables.hpp"

namespace slam {
    namespace traj {
        namespace singer {
            
            // -----------------------------------------------------------------------------
            /**
             * @brief Alias for the constant acceleration trajectory variable.
             *
             * This type alias simplifies access to the `Variable` class within the
             * `slam::traj::const_acc` namespace. The `Variable` represents a state 
             * in a constant acceleration motion model, encapsulating pose, velocity, 
             * and acceleration over time.
             *
             * @see slam::traj::const_acc::Variable
             */
            using Variable = slam::traj::const_acc::Variable;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the process noise covariance matrix for a constant acceleration motion model.
             *
             * This function calculates the **18x18** process noise covariance matrix `Q` for a 
             * given time step `dt`. The function accounts for both large and small acceleration 
             * damping values using an exponential model for high damping values and a Taylor 
             * series expansion for low damping values.
             *
             * @param dt    Time step duration.
             * @param add   Damping coefficient vector (6x1), representing per-axis acceleration damping.
             * @param qcd   Process noise diagonal (6x1), representing noise intensities for each axis. 
             *              Defaults to a unit vector if not specified.
             * @return      **18x18 process noise covariance matrix**.
             *
             * The process noise covariance matrix `Q` is derived based on two cases:
             *  - **High damping regime** (`|add| >= 1.0`): Uses an exponential model.
             *  - **Low damping regime** (`|add| < 1.0`): Uses a Taylor series expansion.
             *
             * **Mathematical Formulation:**
             * \f[
             * Q = \int_0^T \Phi(T, s) G Q_c G^T \Phi(T, s)^T ds
             * \f]
             * where:
             * - `Φ(T, s)` is the state transition matrix,
             * - `G` is the noise injection matrix,
             * - `Q_c` is the continuous-time noise covariance.
             *
             * This formulation ensures proper handling of stochastic noise integration in 
             * constant-acceleration motion models.
             *
             * @note Used in Gaussian Process Regression (GP) priors for factor graph optimization.
             *
             * @see getQinv(), getTran()
             */
            inline Eigen::Matrix<double, 18, 18> getQ(
                const double& dt, const Eigen::Matrix<double, 6, 1>& add,
                const Eigen::Matrix<double, 6, 1>& qcd = Eigen::Matrix<double, 6, 1>::Ones()) {

                Eigen::Matrix<double, 18, 18> Q = Eigen::Matrix<double, 18, 18>::Zero();

                for (int i = 0; i < 6; ++i) {
                    const double ad = add(i);
                    const double qc = qcd(i);

                    if (std::abs(ad) >= 1.0) {
                        const double adi = 1.0 / ad, adi2 = adi * adi, adi3 = adi2 * adi;
                        const double adi4 = adi3 * adi, adi5 = adi4 * adi;
                        const double adt = ad * dt, adt2 = adt * adt, adt3 = adt2 * adt;
                        const double exp_adt = std::exp(-adt), exp_2adt = exp_adt * exp_adt;

                        const double common_qc = 0.5 * qc * adi;
                        const double common_exp = 1 - exp_2adt;
                        
                        Q(i, i) = common_qc * adi4 * (common_exp + 2 * adt - 2 * adt2 - 4 * adt * exp_adt + (2.0 / 3.0) * adt3);
                        Q(i, i + 6) = Q(i + 6, i) = common_qc * adi3 * (exp_2adt + 1 - 2 * exp_adt + 2 * adt * exp_adt - 2 * adt + adt2);
                        Q(i, i + 12) = Q(i + 12, i) = common_qc * adi2 * (common_exp - 2 * adt * exp_adt);
                        Q(i + 6, i + 6) = common_qc * adi2 * (4 * exp_adt - 3 - exp_2adt + 2 * adt);
                        Q(i + 6, i + 12) = Q(i + 12, i + 6) = common_qc * adi * (exp_2adt + 1 - 2 * exp_adt);
                        Q(i + 12, i + 12) = common_qc * common_exp;
                    } 
                    else {
                        const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
                        const double dt5 = dt4 * dt, dt6 = dt5 * dt, dt7 = dt6 * dt;
                        const double dt8 = dt7 * dt, dt9 = dt8 * dt;

                        const double ad2 = ad * ad, ad3 = ad2 * ad, ad4 = ad3 * ad;

                        Q(i, i) = qc * (0.05 * dt5 - 0.0277778 * dt6 * ad + 0.00992063 * dt7 * ad2 - 0.00277778 * dt8 * ad3 + 0.00065586 * dt9 * ad4);
                        Q(i, i + 6) = Q(i + 6, i) = qc * (0.125 * dt4 - 0.0833333 * dt5 * ad + 0.0347222 * dt6 * ad2 - 0.0111111 * dt7 * ad3 + 0.00295139 * dt8 * ad4);
                        Q(i, i + 12) = Q(i + 12, i) = qc * ((1 / 6) * dt3 - (1 / 6) * dt4 * ad + 0.0916667 * dt5 * ad2 - 0.0361111 * dt6 * ad3 + 0.0113095 * dt7 * ad4);
                        Q(i + 6, i + 6) = qc * ((1 / 3) * dt3 - 0.25 * dt4 * ad + 0.116667 * dt5 * ad2 - 0.0416667 * dt6 * ad3 + 0.0123016 * dt7 * ad4);
                        Q(i + 6, i + 12) = Q(i + 12, i + 6) = qc * (0.5 * dt2 - 0.5 * dt3 * ad + 0.291667 * dt4 * ad2 - 0.125 * dt5 * ad3 + 0.0430556 * dt6 * ad4);
                        Q(i + 12, i + 12) = qc * (dt - dt2 * ad + (2 / 3) * dt3 * ad2 - (1 / 3) * dt4 * ad3 + 0.133333 * dt5 * ad4);
                    }
                }

                return Q;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the state transition matrix for a constant acceleration motion model.
             *
             * This function calculates the **18x18** state transition matrix `Φ(dt)`, which models 
             * the evolution of a state vector under constant acceleration with damping.
             *
             * @param dt    Time step duration.
             * @param add   Damping coefficient vector (6x1), representing per-axis acceleration damping.
             * @return      **18x18 state transition matrix**.
             *
             * The transition matrix `Φ(dt)` is computed differently based on the damping regime:
             *  - **High damping regime** (`|add| >= 1.0`): Uses an exponential decay model.
             *  - **Low damping regime** (`|add| < 1.0`): Uses a Taylor series expansion.
             *
             * **Mathematical Formulation:**
             * \f[
             * x(t + dt) = \Phi(dt) \cdot x(t)
             * \f]
             * where `Φ(dt)` is derived from the continuous-time state-space representation:
             * \f[
             * \dot{x} = A x + w
             * \f]
             * and accounts for acceleration damping.
             *
             * **Matrix Structure:**
             * - `Φ(dt)` is **block structured** with the standard time-dependent terms.
             * - The **upper diagonal blocks** encode **position and velocity updates** based on `dt`.
             * - The **lower diagonal blocks** include **damping-dependent exponential or Taylor terms**.
             *
             * @note Used in Gaussian Process Regression (GP) priors for factor graph optimization.
             *
             * @see getQ(), getQinv()
             */
            inline Eigen::Matrix<double, 18, 18> getTran(
                const double& dt, const Eigen::Matrix<double, 6, 1>& add) {

                Eigen::Matrix<double, 18, 18> Tran = Eigen::Matrix<double, 18, 18>::Identity();

                for (int i = 0; i < 6; ++i) {
                    const double ad = add(i);

                    if (std::abs(ad) >= 1.0) {
                        const double ad_inv = 1.0 / ad;
                        const double adt = ad * dt;
                        const double exp_adt = std::exp(-adt);

                        const double ad_inv2 = ad_inv * ad_inv;  // Precompute squared inverse

                        Tran(i, i + 12) = (adt - 1.0 + exp_adt) * ad_inv2;   // C1
                        Tran(i + 6, i + 12) = (1.0 - exp_adt) * ad_inv;      // C2
                        Tran(i + 12, i + 12) = exp_adt;                      // C3
                    } 
                    else {
                        // Use Taylor series expansion for small ad values
                        const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
                        const double dt5 = dt4 * dt, dt6 = dt5 * dt;

                        const double ad2 = ad * ad, ad3 = ad2 * ad, ad4 = ad3 * ad;

                        Tran(i, i + 12) = 0.5 * dt2 - (1.0 / 6.0) * dt3 * ad + (1.0 / 24.0) * dt4 * ad2
                                        - (1.0 / 120.0) * dt5 * ad3 + (1.0 / 720.0) * dt6 * ad4;

                        Tran(i + 6, i + 12) = dt - 0.5 * dt2 * ad + (1.0 / 6.0) * dt3 * ad2
                                            - (1.0 / 24.0) * dt4 * ad3 + (1.0 / 120.0) * dt5 * ad4;

                        Tran(i + 12, i + 12) = 1.0 - dt * ad + 0.5 * dt2 * ad2
                                            - (1.0 / 6.0) * dt3 * ad3 + (1.0 / 24.0) * dt4 * ad4;
                    }
                }

                Tran.block<6, 6>(0, 6).diagonal().array() = dt;  // Assign dt directly

                return Tran;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the Jacobian of the first trajectory knot in a Gaussian process (GP) prior.
             *
             * This function calculates the **18x18 Jacobian matrix** for the first state `knot1` with respect
             * to the **relative transformation**, velocity, and acceleration in a factor graph optimization
             * setting. It accounts for **Lie algebra representation** of SE(3) transformations and applies
             * **Gaussian Process (GP) regression** for motion modeling.
             *
             * @param knot1  Pointer to the first trajectory knot (state).
             * @param knot2  Pointer to the second trajectory knot (state).
             * @param ad     Damping coefficient vector (6x1) affecting acceleration dynamics.
             * @return       **18x18 Jacobian matrix** representing partial derivatives w.r.t. `knot1`.
             *
             * Mathematical Formulation:
             * Given the relative transformation:
             * \f[
             * \xi_{21} = \log(T_2 T_1^{-1})
             * \f]
             * and its associated **inverse right Jacobian**:
             * \f[
             * J_{21}^{-1} = \text{vec2jacinv}(\xi_{21})
             * \f]
             * The Jacobian for `knot1` is computed as:
             * \f[
             * \frac{\partial X}{\partial x_1} = \begin{bmatrix}
             * -J_{21}^{-1} T_{21}^\top & 0 & 0 \\
             * -\frac{1}{2} \hat{w}_2 J_{21}^{-1} & -\Phi_{0,6} & 0 \\
             * -\frac{1}{4} \hat{w}_2^2 J_{21}^{-1} - \frac{1}{2} \hat{a}_2 J_{21}^{-1} & -\Phi_{0,12} & -\Phi_{12,12}
             * \end{bmatrix}
             * \f]
             * where:
             *  - `\hat{w}_2` is the **Lie algebra curly hat operator** for velocity `w_2`.
             *  - `\hat{a}_2` is the **Lie algebra curly hat operator** for acceleration `dw_2`.
             *  - `\Phi(dt, ad)` is the **state transition matrix** capturing damping effects.
             *
             * Matrix Structure:
             * - **Pose Jacobian** (`block<6,6>(0,0) to block<6,6>(12,0)`): Transformation derivatives.
             * - **Velocity Jacobian** (`block<6,6>(0,6) to block<6,6>(12,6)`): Affected by state transition.
             * - **Acceleration Jacobian** (`block<6,6>(0,12) to block<6,6>(12,12)`): Incorporates damping effects.
             *
             * @note Used in **factor graph optimization** and **Gaussian Process Regression (GP)** priors.
             *
             * @see getJacKnot2(), getTran(), getQ()
             */
            inline Eigen::Matrix<double, 18, 18> getJacKnot1(
                const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2,
                const Eigen::Matrix<double, 6, 1>& ad) {

                // Precompute transformations and Jacobians
                const auto T_21 = knot2->getPose()->value() / knot1->getPose()->value();
                const auto xi_21 = T_21.vec();
                const auto J_21_inv = liemath::se3::vec2jacinv(xi_21);
                const double dt = (knot2->getTime() - knot1->getTime()).seconds();
                const auto Phi = getTran(dt, ad);
                const auto Jinv_12 = J_21_inv * T_21.adjoint();
                const auto w2 = knot2->getVelocity()->value();
                const auto dw2 = knot2->getAcceleration()->value();

                // Initialize Jacobian to zero
                Eigen::Matrix<double, 18, 18> jacobian = Eigen::Matrix<double, 18, 18>::Zero();

                // Precompute common Lie algebra terms
                const auto curlyhat_w2 = liemath::se3::curlyhat(w2);
                const auto curlyhat_w2_sq = curlyhat_w2 * curlyhat_w2;
                const auto curlyhat_dw2 = liemath::se3::curlyhat(dw2);

                // Pose Jacobian
                jacobian.block<6, 6>(0, 0) = -Jinv_12;
                jacobian.block<6, 6>(6, 0) = -0.5 * curlyhat_w2 * Jinv_12;
                jacobian.block<6, 6>(12, 0) = -0.25 * curlyhat_w2_sq * Jinv_12 - 0.5 * curlyhat_dw2 * Jinv_12;

                // Velocity Jacobian
                jacobian.block<6, 6>(0, 6) = -Phi.block<6, 6>(0, 6);
                jacobian.block<6, 6>(6, 6) = -Phi.block<6, 6>(6, 6);

                // Acceleration Jacobian
                jacobian.block<6, 6>(0, 12) = -Phi.block<6, 6>(0, 12);
                jacobian.block<6, 6>(6, 12) = -Phi.block<6, 6>(6, 12);
                jacobian.block<6, 6>(12, 12) = -Phi.block<6, 6>(12, 12);

                return jacobian;
            }
        }  // namespace singer
    }  // namespace traj
}  // namespace slam