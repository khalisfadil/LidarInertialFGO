#pragma once

#include <Eigen/Core>
#include <memory>

#include "Trajectory/ConstVelocity/Variables.hpp"
#include "LGMath/LieGroupMath.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {

            // -----------------------------------------------------------------------------
            // Computes the Jacobian "F" associated with the earlier knot.
            inline Eigen::Matrix<double, 12, 12> getJacKnot1(
                const Variable::ConstPtr& knot1, 
                const Variable::ConstPtr& knot2) {
                // Precompute required values
                const auto T_21 = knot2->getPose()->value() / knot1->getPose()->value();
                const auto xi_21 = T_21.vec();
                const auto J_21_inv = slam::liemath::se3::vec2jacinv(xi_21);
                const double dt = (knot2->getTime() - knot1->getTime()).seconds();

                // Explicitly type the matrix multiplication to avoid Eigen vectorization issues
                Eigen::Matrix<double, 6, 6> Jinv_12 = J_21_inv * T_21.adjoint();

                // Initialize Jacobian
                Eigen::Matrix<double, 12, 12> jacobian = Eigen::Matrix<double, 12, 12>::Zero();

                // Pose Jacobians
                jacobian.block<6, 6>(0, 0) = -Jinv_12;
                
                // Split curlyhat operation for clarity
                const auto curlyhat_vel2 = slam::liemath::se3::curlyhat(knot2->getVelocity()->value());
                const auto scaled_curlyhat = -0.5 * curlyhat_vel2;
                jacobian.block<6, 6>(6, 0) = scaled_curlyhat * Jinv_12;

                // Velocity Jacobians
                jacobian.block<6, 6>(0, 6) = -dt * Eigen::Matrix<double, 6, 6>::Identity();
                jacobian.block<6, 6>(6, 6) = -Eigen::Matrix<double, 6, 6>::Identity();

                return jacobian;
            }

            // -----------------------------------------------------------------------------
            // Computes the Jacobian "E" associated with the later knot.
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
                
                // Split curlyhat operation for clarity
                const auto curlyhat_vel2 = slam::liemath::se3::curlyhat(knot2->getVelocity()->value());
                const auto scaled_curlyhat = 0.5 * curlyhat_vel2;
                jacobian.block<6, 6>(6, 0) = scaled_curlyhat * J_21_inv;

                // Velocity Jacobians
                jacobian.block<6, 6>(6, 6) = J_21_inv;

                return jacobian;
            }

            // -----------------------------------------------------------------------------
            // Computes the inverse Jacobian of getJacKnot2.
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
                
                // Split operations for clarity
                const auto curlyhat_vel2 = slam::liemath::se3::curlyhat(knot2->getVelocity()->value());
                const auto J21_curlyhat = J_21 * curlyhat_vel2;
                gamma_inv.block<6, 6>(6, 0) = -0.5 * J21_curlyhat;

                // Velocity Jacobians
                gamma_inv.block<6, 6>(6, 6) = J_21;

                return gamma_inv;
            }

            // -----------------------------------------------------------------------------
            // Computes the Xi transformation matrix.
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
            // Computes the inverse process noise covariance matrix.
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
                
                // Assign off-diagonal terms separately for clarity
                const auto off_diag_term = (-6.0) * dtinv2 * Qcinv_diag;
                Qinv.block<6, 6>(0, 6).diagonal() = off_diag_term;
                Qinv.block<6, 6>(6, 0).diagonal() = off_diag_term;

                return Qinv;
            }

            // -----------------------------------------------------------------------------
            // Computes the process noise covariance matrix.
            inline Eigen::Matrix<double, 12, 12> getQ(
                const double& dt, 
                const Eigen::Matrix<double, 6, 1>& Qc_diag) {
                double dt2 = dt * dt;
                double dt3 = dt * dt2;

                Eigen::Matrix<double, 12, 12> Q = Eigen::Matrix<double, 12, 12>::Zero();
                Q.block<6, 6>(0, 0).diagonal() = dt3 * Qc_diag / 3.0;
                Q.block<6, 6>(6, 6).diagonal() = dt * Qc_diag;
                
                // Assign off-diagonal terms separately for clarity
                const auto off_diag_term = dt2 * Qc_diag / 2.0;
                Q.block<6, 6>(0, 6).diagonal() = off_diag_term;
                Q.block<6, 6>(6, 0).diagonal() = off_diag_term;

                return Q;
            }

            // -----------------------------------------------------------------------------
            // Computes the state transition matrix for constant velocity motion.
            inline Eigen::Matrix<double, 12, 12> getTran(const double& dt) {
                Eigen::Matrix<double, 12, 12> Tran = Eigen::Matrix<double, 12, 12>::Identity();
                Tran.block<6, 6>(0, 6) = Eigen::Matrix<double, 6, 6>::Identity() * dt;
                return Tran;
            }

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam