#include <include/LieGroupMath/so3/Operations.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>

namespace slam {
    namespace liemath {
        namespace so3 {
            
            // -----------------------------------------------------------------------------
            // Converts a 3D vector to a 3x3 skew-symmetric matrix 
            // -----------------------------------------------------------------------------
            
            Eigen::Matrix3d hat(const Eigen::Vector3d& vector) {
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
            // Converts an axis-angle vector to a rotation matrix using the exponential map
            // -----------------------------------------------------------------------------

            Eigen::Matrix3d vec2rot(const Eigen::Vector3d& aaxis_ba, unsigned int numTerms) {
                const double phi_ba = aaxis_ba.norm();
                if (phi_ba < 1e-12) {
                    return Eigen::Matrix3d::Identity();
                }

                if (numTerms == 0) {
                    Eigen::Vector3d axis = aaxis_ba / phi_ba;
                    const double sinphi = std::sin(phi_ba);
                    const double cosphi = std::cos(phi_ba);
                    return cosphi * Eigen::Matrix3d::Identity() +
                        (1.0 - cosphi) * axis * axis.transpose() +
                        sinphi * slam::liemath::so3::hat(axis);
                } else {
                    Eigen::Matrix3d C_ab = Eigen::Matrix3d::Identity();
                    Eigen::Matrix3d x_small = slam::liemath::so3::hat(aaxis_ba);
                    Eigen::Matrix3d x_small_n = Eigen::Matrix3d::Identity();

                    for (unsigned int n = 1; n <= numTerms; ++n) {
                    x_small_n = x_small_n * x_small / static_cast<double>(n);
                    C_ab += x_small_n;
                    }
                    return C_ab;
                }
            }

            // -----------------------------------------------------------------------------
            // Computes both the rotation matrix and its Jacobian
            // -----------------------------------------------------------------------------

            void vec2rot(const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba, 
                        Eigen::Matrix3d* out_C_ab, 
                        Eigen::Matrix3d* out_J_ab) noexcept {
                assert(out_C_ab && "Null pointer out_C_ab in vec2rot");
                assert(out_J_ab && "Null pointer out_J_ab in vec2rot");

                *out_J_ab = slam::liemath::so3::vec2jac(aaxis_ba);
                *out_C_ab = Eigen::Matrix3d::Identity() + slam::liemath::so3::hat(aaxis_ba) * (*out_J_ab);
            }

            // -----------------------------------------------------------------------------
            // Converts a rotation matrix to an axis-angle vector using the logarithm map
            // -----------------------------------------------------------------------------

            Eigen::Vector3d rot2vec(const Eigen::Ref<const Eigen::Matrix3d>& C_ab, const double eps) noexcept {
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
                            return phi_ba * eigenSolver.eigenvectors().col(i);
                        }
                    }
                    throw std::runtime_error("SO(3) logarithmic map failed: no valid axis found.");
                } else {
                    return Eigen::Vector3d::Zero();
                }
            }

            // -----------------------------------------------------------------------------
            // Computes the left Jacobian matrix of SO(3)
            // -----------------------------------------------------------------------------

            Eigen::Matrix3d vec2jac(const Eigen::Vector3d& aaxis_ba, unsigned int numTerms) {
                const double phi_ba = aaxis_ba.norm();
                if (phi_ba < 1e-12) {
                    return Eigen::Matrix3d::Identity();
                }

                if (numTerms == 0) {
                    Eigen::Vector3d axis = aaxis_ba / phi_ba;
                    const double sinTerm = std::sin(phi_ba) / phi_ba;
                    const double cosTerm = (1.0 - std::cos(phi_ba)) / phi_ba;
                    return sinTerm * Eigen::Matrix3d::Identity() +
                        (1.0 - sinTerm) * axis * axis.transpose() +
                        cosTerm * slam::liemath::so3::hat(axis);
                } else {
                    Eigen::Matrix3d J_ab = Eigen::Matrix3d::Identity();
                    Eigen::Matrix3d x_small = slam::liemath::so3::hat(aaxis_ba);
                    Eigen::Matrix3d x_small_n = Eigen::Matrix3d::Identity();

                    for (unsigned int n = 1; n <= numTerms; ++n) {
                    x_small_n = x_small_n * x_small / static_cast<double>(n + 1);
                    J_ab += x_small_n;
                    }
                    return J_ab;
                }
            }

            // -----------------------------------------------------------------------------
            // Computes the inverse of the SO(3) left Jacobian matrix
            // -----------------------------------------------------------------------------
            
            Eigen::Matrix3d vec2jacinv(const Eigen::Vector3d& aaxis_ba, unsigned int numTerms) {
                const double phi_ba = aaxis_ba.norm();
                if (phi_ba < 1e-12) {
                    return Eigen::Matrix3d::Identity();
                }

                if (numTerms == 0) {
                    Eigen::Vector3d axis = aaxis_ba / phi_ba;
                    const double halfphi = 0.5 * phi_ba;
                    const double cotanTerm = halfphi / std::tan(halfphi);
                    return cotanTerm * Eigen::Matrix3d::Identity() +
                        (1.0 - cotanTerm) * axis * axis.transpose() -
                        halfphi * hat(axis);
                } else {
                    if (numTerms > 20) {
                        throw std::invalid_argument("Numerical vec2jacinv does not support numTerms > 20");
                    }

                    Eigen::Matrix3d J_ab_inverse = Eigen::Matrix3d::Identity();
                    Eigen::Matrix3d x_small = slam::liemath::so3::hat(aaxis_ba);
                    Eigen::Matrix3d x_small_n = Eigen::Matrix3d::Identity();

                    static const Eigen::Matrix<double, 21, 1> bernoulli = (Eigen::Matrix<double, 21, 1>() <<
                        1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0,
                        -1.0 / 30.0, 0.0, 5.0 / 66.0, 0.0, -691.0 / 2730.0, 0.0, 7.0 / 6.0, 0.0,
                        -3617.0 / 510.0, 0.0, 43867.0 / 798.0, 0.0, -174611.0 / 330.0).finished();

                    for (unsigned int n = 1; n <= numTerms; ++n) {
                        x_small_n = x_small_n * x_small / static_cast<double>(n);
                        J_ab_inverse += bernoulli(n) * x_small_n;
                    }
                    return J_ab_inverse;
                }
            }
        }  // namespace so3
    } // namespace liemath
}  // namespace slam
