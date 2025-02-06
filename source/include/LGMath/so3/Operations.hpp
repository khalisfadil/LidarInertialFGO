#pragma once

#include <Eigen/Core>

/// Special Orthogonal Group (SO3) Lie Group Mathematics
namespace slam {
    namespace liemath {
        namespace so3 {
            
            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs a 3x3 skew-symmetric matrix (hat operator) from a 3x1 vector.
             * @param vector 3x1 vector.
             * @return 3x3 skew-symmetric matrix.
             */
            Eigen::Matrix3d hat(const Eigen::Vector3d& vector);

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes a rotation matrix using the exponential map.
             * @param aaxis_ba 3x1 axis-angle vector.
             * @param numTerms Number of terms in the Taylor series expansion (default = 0 for analytical solution).
             * @return 3x3 rotation matrix.
             */
            Eigen::Matrix3d vec2rot(const Eigen::Vector3d& aaxis_ba, unsigned int numTerms = 0);

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes both the rotation matrix and its Jacobian.
             * @param aaxis_ba 3x1 axis-angle vector.
             * @param out_C_ab Output rotation matrix.
             * @param out_J_ab Output SO(3) Jacobian matrix.
             */
            void vec2rot(const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba, Eigen::Matrix3d* out_C_ab, Eigen::Matrix3d* out_J_ab) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the logarithmic map (inverse of exponential map) for an SO(3) rotation matrix.
             * @param C_ab 3x3 rotation matrix.
             * @param eps Numerical stability threshold (default = 1e-6).
             * @return 3x1 axis-angle representation.
             */
            Eigen::Vector3d rot2vec(const Eigen::Ref<const Eigen::Matrix3d>& C_ab, const double eps = 1e-6) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the 3x3 left Jacobian of SO(3).
             * @param aaxis_ba 3x1 axis-angle vector.
             * @param numTerms Number of terms in the Taylor series expansion (default = 0 for analytical solution).
             * @return 3x3 Jacobian matrix.
             */
            Eigen::Matrix3d vec2jac(const Eigen::Vector3d& aaxis_ba, unsigned int numTerms = 0);

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the 3x3 inverse left Jacobian of SO(3).
             * @param aaxis_ba 3x1 axis-angle vector.
             * @param numTerms Number of terms in the Taylor series expansion (default = 0 for analytical solution).
             * @return 3x3 inverse Jacobian matrix.
             */
            Eigen::Matrix3d vec2jacinv(const Eigen::Vector3d& aaxis_ba, unsigned int numTerms = 0);

        }  // namespace so3
    } // liemath
}  // namespace slam
