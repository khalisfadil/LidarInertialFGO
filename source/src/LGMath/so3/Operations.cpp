#include "LGMath/so3/Operations.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

namespace slam {
    namespace liemath {
        namespace so3 {

            // -----------------------------------------------------------------------------
            // SO(3) Hat Operator
            // -----------------------------------------------------------------------------

            Eigen::Matrix3d hat(const Eigen::Ref<const Eigen::Vector3d>& vector) noexcept {
                Eigen::Matrix3d mat = Eigen::Matrix3d::Zero();
                mat(0, 1) = -vector[2];
                mat(0, 2) = vector[1];
                mat(1, 0) = vector[2];
                mat(1, 2) = -vector[0];
                mat(2, 0) = -vector[1];
                mat(2, 1) = vector[0];
                return mat;
            }

            // -----------------------------------------------------------------------------
            // SO(3) Exponential Map (Vector to Rotation)
            // -----------------------------------------------------------------------------

            Eigen::Matrix3d vec2rot(const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba, unsigned int numTerms) noexcept {
                const double phi_ba = aaxis_ba.norm();
                if (phi_ba < 1e-12) {
                    return Eigen::Matrix3d::Identity();
                }

                if (numTerms == 0) {
                    const double sinphi = std::sin(phi_ba);
                    const double cosphi = std::cos(phi_ba);
                    Eigen::Vector3d axis = aaxis_ba / phi_ba;
                    Eigen::Matrix3d axis_outer = axis * axis.transpose();
                    return cosphi * Eigen::Matrix3d::Identity() +
                        (1.0 - cosphi) * axis_outer +
                        sinphi * hat(axis);
                }

                Eigen::Matrix3d C_ab = Eigen::Matrix3d::Identity();
                Eigen::Matrix3d x_small = hat(aaxis_ba);
                Eigen::Matrix3d x_small_n = Eigen::Matrix3d::Identity();
                for (unsigned int n = 1; n <= numTerms; ++n) {
                    x_small_n = x_small_n * x_small / static_cast<double>(n);
                    C_ab += x_small_n;
                }
                return C_ab;
            }

            void vec2rot(const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba,
                        Eigen::Matrix3d* out_C_ab,
                        Eigen::Matrix3d* out_J_ab) noexcept {
                assert(out_C_ab && "Null pointer out_C_ab in vec2rot");
                assert(out_J_ab && "Null pointer out_J_ab in vec2rot");

                *out_J_ab = vec2jac(aaxis_ba);
                *out_C_ab = Eigen::Matrix3d::Identity() + hat(aaxis_ba) * (*out_J_ab);
            }

            // -----------------------------------------------------------------------------
            // SO(3) Logarithmic Map (Rotation to Vector)
            // -----------------------------------------------------------------------------

            Eigen::Vector3d rot2vec(const Eigen::Ref<const Eigen::Matrix3d>& C_ab, double eps) noexcept {
                const double trace_term = 0.5 * (C_ab.trace() - 1.0);
                const double phi_ba = std::acos(std::clamp(trace_term, -1.0, 1.0));
                const double sinphi_ba = std::sin(phi_ba);

                if (std::fabs(sinphi_ba) > eps) {
                    Eigen::Vector3d axis;
                    axis << C_ab(2, 1) - C_ab(1, 2),
                            C_ab(0, 2) - C_ab(2, 0),
                            C_ab(1, 0) - C_ab(0, 1);
                    return (0.5 * phi_ba / sinphi_ba) * axis;
                } else if (std::fabs(phi_ba) > eps) {
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(C_ab);
                    for (int i = 0; i < 3; ++i) {
                        if (std::fabs(eigenSolver.eigenvalues()[i] - 1.0) < 1e-6) {
                            return phi_ba * eigenSolver.eigenvectors().col(i).normalized();
                        }
                    }
                    std::cerr << "SO(3) logarithmic map failed: no valid axis found, returning zero" << std::endl;
                    return Eigen::Vector3d::Zero();
                }
                return Eigen::Vector3d::Zero();
            }

            // -----------------------------------------------------------------------------
            // SO(3) Left Jacobian
            // -----------------------------------------------------------------------------

            Eigen::Matrix3d vec2jac(const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba, unsigned int numTerms) noexcept {
                const double phi_ba = aaxis_ba.norm();
                if (phi_ba < 1e-12) {
                    return Eigen::Matrix3d::Identity();
                }

                if (numTerms == 0) {
                    const double sinphi = std::sin(phi_ba);
                    const double cosphi = std::cos(phi_ba);
                    const double sinTerm = sinphi / phi_ba;
                    const double cosTerm = (1.0 - cosphi) / phi_ba;
                    Eigen::Vector3d axis = aaxis_ba / phi_ba;
                    Eigen::Matrix3d axis_outer = axis * axis.transpose();
                    return sinTerm * Eigen::Matrix3d::Identity() +
                        (1.0 - sinTerm) * axis_outer +
                        cosTerm * hat(axis);
                }

                Eigen::Matrix3d J_ab = Eigen::Matrix3d::Identity();
                Eigen::Matrix3d x_small = hat(aaxis_ba);
                Eigen::Matrix3d x_small_n = Eigen::Matrix3d::Identity();
                for (unsigned int n = 1; n <= numTerms; ++n) {
                    x_small_n = x_small_n * x_small / static_cast<double>(n + 1);
                    J_ab += x_small_n;
                }
                return J_ab;
            }

            // -----------------------------------------------------------------------------
            // SO(3) Inverse Left Jacobian
            // -----------------------------------------------------------------------------

            Eigen::Matrix3d vec2jacinv(const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba, unsigned int numTerms) noexcept {
                const double phi_ba = aaxis_ba.norm();
                if (phi_ba < 1e-12) {
                    return Eigen::Matrix3d::Identity();
                }

                if (numTerms == 0) {
                    const double halfphi = 0.5 * phi_ba;
                    const double cotanTerm = halfphi / std::tan(halfphi);
                    Eigen::Vector3d axis = aaxis_ba / phi_ba;
                    Eigen::Matrix3d axis_outer = axis * axis.transpose();
                    return cotanTerm * Eigen::Matrix3d::Identity() +
                        (1.0 - cotanTerm) * axis_outer -
                        halfphi * hat(axis);
                }

                if (numTerms > 20) {
                    std::cerr << "Numerical vec2jacinv: numTerms > 20 not supported, returning identity" << std::endl;
                    return Eigen::Matrix3d::Identity();
                }

                static const double bernoulli[] = {1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0,
                                                -1.0 / 30.0, 0.0, 5.0 / 66.0, 0.0, -691.0 / 2730.0, 0.0, 7.0 / 6.0, 0.0,
                                                -3617.0 / 510.0, 0.0, 43867.0 / 798.0, 0.0, -174611.0 / 330.0};

                Eigen::Matrix3d J_ab_inv = Eigen::Matrix3d::Identity();
                Eigen::Matrix3d x_small = hat(aaxis_ba);
                Eigen::Matrix3d x_small_n = Eigen::Matrix3d::Identity();
                for (unsigned int n = 1; n <= numTerms; ++n) {
                    x_small_n = x_small_n * x_small / static_cast<double>(n);
                    J_ab_inv += bernoulli[n] * x_small_n;
                }
                return J_ab_inv;
            }

        }  // namespace so3
    }  // namespace liemath
}  // namespace slam