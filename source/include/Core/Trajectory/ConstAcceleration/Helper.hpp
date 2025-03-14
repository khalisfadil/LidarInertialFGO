#pragma once

#include <Eigen/Core>
#include "Core/Trajectory/ConstAcceleration/Variables.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the inverse of the process noise covariance matrix Q.
             *
             * Uses precomputed **diagonal inverse elements** to avoid redundant divisions.
             *
             * @param dt        Time step.
             * @param Qc_diag   Diagonal elements of continuous-time process noise.
             * @return          18x18 inverse covariance matrix.
             */
            inline Eigen::Matrix<double, 18, 18> getQinv(
                const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) {

                // Precompute dt inverse powers
                const double dtinv = 1.0 / dt;
                const double dtinv2 = dtinv * dtinv, dtinv3 = dtinv2 * dtinv;
                const double dtinv4 = dtinv3 * dtinv, dtinv5 = dtinv4 * dtinv;

                // Compute inverse of Qc_diag
                const Eigen::Matrix<double, 6, 1> Qcinv_diag = Qc_diag.cwiseInverse();

                // Initialize and set diagonal blocks
                Eigen::Matrix<double, 18, 18> Qinv = Eigen::Matrix<double, 18, 18>::Zero();
                
                Qinv.block<6, 6>(0, 0).diagonal().array() = 720.0 * dtinv5 * Qcinv_diag.array();
                Qinv.block<6, 6>(6, 6).diagonal().array() = 192.0 * dtinv3 * Qcinv_diag.array();
                Qinv.block<6, 6>(12, 12).diagonal().array() = 9.0 * dtinv * Qcinv_diag.array();

                // Fill symmetric off-diagonal blocks efficiently
                const Eigen::Matrix<double, 6, 1> q01 = -360.0 * dtinv4 * Qcinv_diag;
                const Eigen::Matrix<double, 6, 1> q02 = 60.0 * dtinv3 * Qcinv_diag;
                const Eigen::Matrix<double, 6, 1> q12 = -36.0 * dtinv2 * Qcinv_diag;

                Qinv.block<6, 6>(0, 6).diagonal() = Qinv.block<6, 6>(6, 0).diagonal() = q01;
                Qinv.block<6, 6>(0, 12).diagonal() = Qinv.block<6, 6>(12, 0).diagonal() = q02;
                Qinv.block<6, 6>(6, 12).diagonal() = Qinv.block<6, 6>(12, 6).diagonal() = q12;

                return Qinv;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the process noise covariance matrix Q.
             *
             * Uses time step powers for efficiency.
             *
             * @param dt        Time step.
             * @param Qc_diag   Diagonal elements of continuous-time process noise.
             * @return          18x18 covariance matrix.
             */
            inline Eigen::Matrix<double, 18, 18> getQ(
                const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) {

                // Precompute dt powers
                const double dt2 = dt * dt;
                const double dt3 = dt * dt2;
                const double dt4 = dt * dt3;
                const double dt5 = dt * dt4;

                // Initialize Q matrix
                Eigen::Matrix<double, 18, 18> Q = Eigen::Matrix<double, 18, 18>::Zero();

                // Precompute scaled Qc_diag values
                const Eigen::Matrix<double, 6, 1> q00 = (dt5 / 20.0) * Qc_diag;
                const Eigen::Matrix<double, 6, 1> q11 = (dt3 / 3.0) * Qc_diag;
                const Eigen::Matrix<double, 6, 1> q22 = dt * Qc_diag;
                const Eigen::Matrix<double, 6, 1> q01 = (dt4 / 8.0) * Qc_diag;
                const Eigen::Matrix<double, 6, 1> q02 = (dt3 / 6.0) * Qc_diag;
                const Eigen::Matrix<double, 6, 1> q12 = (dt2 / 2.0) * Qc_diag;

                // Assign diagonal values
                Q.block<6, 6>(0, 0).diagonal() = q00;
                Q.block<6, 6>(6, 6).diagonal() = q11;
                Q.block<6, 6>(12, 12).diagonal() = q22;

                // Assign symmetric off-diagonal values
                Q.block<6, 6>(0, 6).diagonal() = Q.block<6, 6>(6, 0).diagonal() = q01;
                Q.block<6, 6>(0, 12).diagonal() = Q.block<6, 6>(12, 0).diagonal() = q02;
                Q.block<6, 6>(6, 12).diagonal() = Q.block<6, 6>(12, 6).diagonal() = q12;

                return Q;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the state transition matrix.
             *
             * @param dt    Time step.
             * @return      18x18 transition matrix.
             */
            inline Eigen::Matrix<double, 18, 18> getTran(const double& dt) {
                Eigen::Matrix<double, 18, 18> Tran = Eigen::Matrix<double, 18, 18>::Identity();

                // Precompute dt terms
                const double dt2 = 0.5 * dt * dt;

                // Assign transformation blocks
                const Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();
                Tran.block<6, 6>(0, 6) = dt * I;
                Tran.block<6, 6>(6, 12) = dt * I;
                Tran.block<6, 6>(0, 12) = dt2 * I;

                return Tran;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the Jacobian with respect to the first knot.
             *
             * Corresponds to matrix **F** in "State Estimation" (2nd Ed) and Tim Tang's thesis.
             *
             * @param knot1    First (earlier) state.
             * @param knot2    Second (later) state.
             * @return         18x18 Jacobian matrix.
             */
            inline Eigen::Matrix<double, 18, 18> getJacKnot1(
                const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) {

                // Precompute transformations and Jacobians
                const auto T_21 = knot2->getPose()->value() / knot1->getPose()->value();
                const auto xi_21 = T_21.vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);
                const Eigen::Matrix<double, 6, 6> Jinv_12 = J_21_inv * T_21.adjoint();

                // Compute time difference and state transition matrix
                const double dt = (knot2->getTime() - knot1->getTime()).seconds();
                const Eigen::Matrix<double, 18, 18> Phi = getTran(dt);

                // Retrieve state values
                const auto w2 = knot2->getVelocity()->value();
                const auto dw2 = knot2->getAcceleration()->value();

                // Precompute commonly used terms
                const Eigen::Matrix<double, 6, 6> curly_w2 = liemath::se3::curlyhat(w2);
                const Eigen::Matrix<double, 6, 6> curly_dw2 = liemath::se3::curlyhat(dw2);
                const Eigen::Matrix<double, 6, 6> curly_w2_sq = curly_w2 * curly_w2;

                // Initialize and assign Jacobian matrix
                Eigen::Matrix<double, 18, 18> jacobian;
                jacobian.setZero();

                // Pose Jacobian
                jacobian.block<6, 6>(0, 0) = -Jinv_12;
                jacobian.block<6, 6>(6, 0) = -0.5 * curly_w2 * Jinv_12;
                jacobian.block<6, 6>(12, 0) = -0.25 * curly_w2_sq * Jinv_12 - 0.5 * curly_dw2 * Jinv_12;

                // Velocity Jacobian
                jacobian.block<6, 6>(0, 6) = jacobian.block<6, 6>(6, 6) = -Phi.block<6, 6>(0, 6);
                jacobian.block<6, 6>(12, 6) = Eigen::Matrix<double, 6, 6>::Zero();

                // Acceleration Jacobian
                jacobian.block<6, 6>(0, 12) = -Phi.block<6, 6>(0, 12);
                jacobian.block<6, 6>(6, 12) = -Phi.block<6, 6>(6, 12);
                jacobian.block<6, 6>(12, 12) = -Phi.block<6, 6>(12, 12);

                return jacobian;
            }

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the Jacobian with respect to the second knot.
             *
             * Corresponds to matrix **E** in "State Estimation" (2nd Ed) and Tim Tang's thesis.
             *
             * @param knot1    First (earlier) state.
             * @param knot2    Second (later) state.
             * @return         18x18 Jacobian matrix.
             */
            inline Eigen::Matrix<double, 18, 18> getJacKnot2(
                const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) {

                // Compute transformation and Jacobians
                const auto T_21 = knot2->getPose()->value() / knot1->getPose()->value();
                const auto xi_21 = T_21.vec();
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);

                // Retrieve state values
                const auto w2 = knot2->getVelocity()->value();
                const auto dw2 = knot2->getAcceleration()->value();

                // Precompute commonly used terms
                const Eigen::Matrix<double, 6, 6> curly_w2 = liemath::se3::curlyhat(w2);
                const Eigen::Matrix<double, 6, 6> curly_dw2 = liemath::se3::curlyhat(dw2);
                const Eigen::Matrix<double, 6, 6> curly_w2_sq = curly_w2 * curly_w2;
                const Eigen::Matrix<double, 6, 6> curly_w2_inv = liemath::se3::curlyhat(J_21_inv * w2);

                // Initialize and assign Jacobian matrix
                Eigen::Matrix<double, 18, 18> jacobian;
                jacobian.setZero();

                // Pose Jacobian
                jacobian.block<6, 6>(0, 0) = J_21_inv;
                jacobian.block<6, 6>(6, 0) = 0.5 * curly_w2 * J_21_inv;
                jacobian.block<6, 6>(12, 0) = 0.25 * curly_w2_sq * J_21_inv + 0.5 * curly_dw2 * J_21_inv;

                // Velocity Jacobian
                jacobian.block<6, 6>(6, 6) = J_21_inv;
                jacobian.block<6, 6>(12, 6) = -0.5 * curly_w2_inv + 0.5 * curly_w2 * J_21_inv;

                // Acceleration Jacobian
                jacobian.block<6, 6>(12, 12) = J_21_inv;

                return jacobian;
            }
        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
