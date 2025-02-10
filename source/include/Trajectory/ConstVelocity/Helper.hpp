#pragma once

#include <Eigen/Core>
#include <memory>

#include "source/include/Trajectory/ConstVelocity/Variables.hpp"
#include "source/include/LGMath/LieGroupMath.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the Jacobian "F" associated with the earlier knot.
             * @details See State Estimation (2nd Ed) Section 11.1.4 for details.
             * 
             * @param knot1 First control point (earlier in time).
             * @param knot2 Second control point (later in time).
             * @return 12x12 Jacobian matrix.
             */
            inline Eigen::Matrix<double, 12, 12> getJacKnot1(
                const Variable::ConstPtr& knot1, 
                const Variable::ConstPtr& knot2) {

                // Precompute required values
                const auto T_21 = knot2->getPose()->value() / knot1->getPose()->value();
                const auto xi_21 = T_21.vec();
                const auto J_21_inv = slam::liemath::se3::vec2jacinv(xi_21);
                const double dt = (knot2->getTime() - knot1->getTime()).seconds();

                const auto Jinv_12 = J_21_inv * T_21.adjoint();

                // Initialize Jacobian
                Eigen::Matrix<double, 12, 12> jacobian = Eigen::Matrix<double, 12, 12>::Zero();

                // Pose Jacobians
                jacobian.block<6, 6>(0, 0) = -Jinv_12;
                jacobian.block<6, 6>(6, 0) = 
                    -0.5 * slam::liemath::se3::curlyhat(knot2->getVelocity()->value()) * Jinv_12;

                // Velocity Jacobians
                jacobian.block<6, 6>(0, 6) = -dt * Eigen::Matrix<double, 6, 6>::Identity();
                jacobian.block<6, 6>(6, 6) = -Eigen::Matrix<double, 6, 6>::Identity();

                return jacobian;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the Jacobian "E" associated with the later knot.
             * @param knot1 First control point.
             * @param knot2 Second control point.
             * @return 12x12 Jacobian matrix.
             */
            inline Eigen::Matrix<double, 12, 12> getJacKnot2(
                const Variable::ConstPtr& knot1, 
                const Variable::ConstPtr& knot2) {

                // Precompute required values
                const auto T_21 = knot2->getPose()->value() / knot1->getPose()->value();
                const auto xi_21 = T_21.vec();
                const auto J_21_inv = slam::liemath::se3::vec2jacinv(xi_21);

                // Initialize Jacobian
                Eigen::Matrix<double, 12, 12> jacobian = Eigen::Matrix<double, 12, 12>::Zero();

                // Pose Jacobians
                jacobian.block<6, 6>(0, 0) = J_21_inv;
                jacobian.block<6, 6>(6, 0) = 
                    0.5 * slam::liemath::se3::curlyhat(knot2->getVelocity()->value()) * J_21_inv;

                // Velocity Jacobians
                jacobian.block<6, 6>(6, 6) = J_21_inv;

                return jacobian;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the inverse Jacobian of getJacKnot2.
             * @param knot1 First control point.
             * @param knot2 Second control point.
             * @return 12x12 Jacobian inverse matrix.
             */
            inline Eigen::Matrix<double, 12, 12> getJacKnot3(
                const Variable::ConstPtr& knot1, 
                const Variable::ConstPtr& knot2) {

                // Precompute required values
                const auto T_21 = knot2->getPose()->value() / knot1->getPose()->value();
                const auto xi_21 = T_21.vec();
                const auto J_21 = slam::liemath::se3::vec2jac(xi_21);

                // Initialize Jacobian inverse
                Eigen::Matrix<double, 12, 12> gamma_inv = Eigen::Matrix<double, 12, 12>::Zero();

                // Pose Jacobians
                gamma_inv.block<6, 6>(0, 0) = J_21;
                gamma_inv.block<6, 6>(6, 0) = 
                    -0.5 * J_21 * slam::liemath::se3::curlyhat(knot2->getVelocity()->value());

                // Velocity Jacobians
                gamma_inv.block<6, 6>(6, 6) = J_21;

                return gamma_inv;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the Xi transformation matrix.
             * @param knot1 First control point.
             * @param knot2 Second control point.
             * @return 12x12 Xi matrix.
             */
            inline Eigen::Matrix<double, 12, 12> getXi(
                const Variable::ConstPtr& knot1, 
                const Variable::ConstPtr& knot2) {

                const auto T_21 = knot2->getPose()->value() / knot1->getPose()->value();
                const auto Tau_21 = T_21.adjoint();

                Eigen::Matrix<double, 12, 12> Xi = Eigen::Matrix<double, 12, 12>::Zero();
                Xi.block<6, 6>(0, 0) = Tau_21;

                return Xi;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the inverse process noise covariance matrix.
             * @param dt Time step.
             * @param Qc_diag Diagonal elements of continuous-time process noise covariance.
             * @return 12x12 inverse process noise covariance matrix.
             */
            inline Eigen::Matrix<double, 12, 12> getQinv(
                const double& dt, 
                const Eigen::Matrix<double, 6, 1>& Qc_diag) {

                Eigen::Matrix<double, 6, 1> Qcinv_diag = 1.0 / Qc_diag.array();
                double dtinv = 1.0 / dt;
                double dtinv2 = dtinv * dtinv;
                double dtinv3 = dtinv * dtinv2;

                Eigen::Matrix<double, 12, 12> Qinv = Eigen::Matrix<double, 12, 12>::Zero();
                Qinv.block<6, 6>(0, 0).diagonal() = 12.0 * dtinv3 * Qcinv_diag;
                Qinv.block<6, 6>(6, 6).diagonal() = 4.0 * dtinv * Qcinv_diag;
                Qinv.block<6, 6>(0, 6).diagonal() = Qinv.block<6, 6>(6, 0).diagonal() = (-6.0) * dtinv2 * Qcinv_diag;

                return Qinv;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the process noise covariance matrix.
             * @param dt Time step.
             * @param Qc_diag Diagonal elements of continuous-time process noise covariance.
             * @return 12x12 process noise covariance matrix.
             */
            inline Eigen::Matrix<double, 12, 12> getQ(
                const double& dt, 
                const Eigen::Matrix<double, 6, 1>& Qc_diag) {

                double dt2 = dt * dt;
                double dt3 = dt * dt2;

                Eigen::Matrix<double, 12, 12> Q = Eigen::Matrix<double, 12, 12>::Zero();
                Q.block<6, 6>(0, 0).diagonal() = dt3 * Qc_diag / 3.0;
                Q.block<6, 6>(6, 6).diagonal() = dt * Qc_diag;
                Q.block<6, 6>(0, 6).diagonal() = Q.block<6, 6>(6, 0).diagonal() = dt2 * Qc_diag / 2.0;

                return Q;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the state transition matrix for constant velocity motion.
             *
             * This function constructs a **12×12** transition matrix used in constant velocity
             * motion models. It represents the integration of velocity into position over
             * a time interval `dt`.
             *
             * The state vector consists of:
             * - **First 6 elements**: Position (or pose in SE(3))
             * - **Last 6 elements**: Velocity (Lie algebra representation)
             *
             * The transition matrix follows:
             * \f[
             *   \mathbf{T} =
             *   \begin{bmatrix}
             *     \mathbf{I} & \Delta t \cdot \mathbf{I} \\
             *     \mathbf{0} & \mathbf{I}
             *   \end{bmatrix}
             * \f]
             *
             * @param dt Time step for integration.
             * @return 12×12 state transition matrix.
             */
            inline Eigen::Matrix<double, 12, 12> getTran(const double& dt) {
                Eigen::Matrix<double, 12, 12> Tran = Eigen::Matrix<double, 12, 12>::Identity();
                Tran.block<6, 6>(0, 6) = Eigen::Matrix<double, 6, 6>::Identity() * dt;
                return Tran;
            }

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
