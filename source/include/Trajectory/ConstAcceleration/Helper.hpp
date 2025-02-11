#pragma once

#include <Eigen/Core>
#include "source/include/Trajectory/ConstAcceleration/Variables.hpp"

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

                Eigen::Matrix<double, 6, 1> Qcinv_diag = 1.0 / Qc_diag.array();
                const double dtinv = 1.0 / dt, dtinv2 = dtinv * dtinv, dtinv3 = dtinv * dtinv2;
                const double dtinv4 = dtinv * dtinv3, dtinv5 = dtinv * dtinv4;

                Eigen::Matrix<double, 18, 18> Qinv = Eigen::Matrix<double, 18, 18>::Zero();
                Qinv.block<6, 6>(0, 0).diagonal() = 720.0 * dtinv5 * Qcinv_diag;
                Qinv.block<6, 6>(6, 6).diagonal() = 192.0 * dtinv3 * Qcinv_diag;
                Qinv.block<6, 6>(12, 12).diagonal() = 9.0 * dtinv * Qcinv_diag;
                Qinv.block<6, 6>(0, 6).diagonal() = Qinv.block<6, 6>(6, 0).diagonal() = (-360.0) * dtinv4 * Qcinv_diag;
                Qinv.block<6, 6>(0, 12).diagonal() = Qinv.block<6, 6>(12, 0).diagonal() = (60.0) * dtinv3 * Qcinv_diag;
                Qinv.block<6, 6>(6, 12).diagonal() = Qinv.block<6, 6>(12, 6).diagonal() = (-36.0) * dtinv2 * Qcinv_diag;
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

                const double dt2 = dt * dt, dt3 = dt * dt2, dt4 = dt * dt3, dt5 = dt * dt4;

                Eigen::Matrix<double, 18, 18> Q = Eigen::Matrix<double, 18, 18>::Zero();
                Q.block<6, 6>(0, 0).diagonal() = dt5 * Qc_diag / 20.0;
                Q.block<6, 6>(6, 6).diagonal() = dt3 * Qc_diag / 3.0;
                Q.block<6, 6>(12, 12).diagonal() = dt * Qc_diag;
                Q.block<6, 6>(0, 6).diagonal() = Q.block<6, 6>(6, 0).diagonal() = dt4 * Qc_diag / 8.0;
                Q.block<6, 6>(0, 12).diagonal() = Q.block<6, 6>(12, 0).diagonal() = dt3 * Qc_diag / 6.0;
                Q.block<6, 6>(6, 12).diagonal() = Q.block<6, 6>(12, 6).diagonal() = dt2 * Qc_diag / 2.0;
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
                const auto I = Eigen::Matrix<double, 6, 6>::Identity();
                Tran.block<6, 6>(0, 6) = Tran.block<6, 6>(6, 12) = dt * I;
                Tran.block<6, 6>(0, 12) = 0.5 * dt * dt * I;
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

                const auto T_21 = knot2->pose()->value() / knot1->pose()->value();
                const auto xi_21 = T_21.vec();
                const auto J_21_inv = liemath::se3::vec2jacinv(xi_21);
                const double dt = (knot2->time() - knot1->time()).seconds();
                const Eigen::Matrix<double, 18, 18> Phi = getTran(dt);
                const auto Jinv_12 = J_21_inv * T_21.adjoint();

                const auto w2 = knot2->velocity()->value();
                const auto dw2 = knot2->acceleration()->value();

                Eigen::Matrix<double, 18, 18> jacobian = Eigen::Matrix<double, 18, 18>::Zero();
                jacobian.block<6, 6>(0, 0) = -Jinv_12;
                jacobian.block<6, 6>(6, 0) = -0.5 * liemath::se3::curlyhat(w2) * Jinv_12;
                jacobian.block<6, 6>(12, 0) = -0.25 * liemath::se3::curlyhat(w2) * liemath::se3::curlyhat(w2) * Jinv_12 - 
                                            0.5 * liemath::se3::curlyhat(dw2) * Jinv_12;
                jacobian.block<6, 6>(0, 6) = jacobian.block<6, 6>(6, 6) = -Phi.block<6, 6>(0, 6);
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

                const auto T_21 = knot2->pose()->value() / knot1->pose()->value();
                const auto xi_21 = T_21.vec();
                const auto J_21_inv = liemath::se3::vec2jacinv(xi_21);
                const auto w2 = knot2->velocity()->value();
                const auto dw2 = knot2->acceleration()->value();

                Eigen::Matrix<double, 18, 18> jacobian = Eigen::Matrix<double, 18, 18>::Zero();
                jacobian.block<6, 6>(0, 0) = J_21_inv;
                jacobian.block<6, 6>(6, 0) = 0.5 * liemath::se3::curlyhat(w2) * J_21_inv;
                jacobian.block<6, 6>(12, 12) = J_21_inv;
                return jacobian;
            }
        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
