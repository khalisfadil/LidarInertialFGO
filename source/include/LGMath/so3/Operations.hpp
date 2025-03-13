#pragma once

#include <Eigen/Dense>

namespace slam {
    namespace liemath {
        namespace so3 {

            // -----------------------------------------------------------------------------
            // SO(3) Hat Operator
            // -----------------------------------------------------------------------------

            /**
             * @brief Constructs a 3x3 skew-symmetric matrix (hat operator) from a 3x1 vector.
             * @param vector 3x1 vector.
             * @return 3x3 skew-symmetric matrix.
             */
            Eigen::Matrix3d hat(const Eigen::Ref<const Eigen::Vector3d>& vector) noexcept;

            // -----------------------------------------------------------------------------
            // SO(3) Exponential Map (Vector to Rotation)
            // -----------------------------------------------------------------------------

            /**
             * @brief Computes a 3x3 rotation matrix from an axis-angle vector using the exponential map.
             * @param aaxis_ba 3x1 axis-angle vector.
             * @param numTerms Number of terms in the Taylor series (0 = analytical, >0 = numerical).
             * @return 3x3 rotation matrix.
             */
            Eigen::Matrix3d vec2rot(const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba, unsigned int numTerms = 0) noexcept;

            /**
             * @brief Computes the rotation matrix and its Jacobian from an axis-angle vector.
             * @param aaxis_ba 3x1 axis-angle vector.
             * @param out_C_ab Output pointer to 3x3 rotation matrix (must not be null).
             * @param out_J_ab Output pointer to 3x3 Jacobian matrix (must not be null).
             */
            void vec2rot(const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba,
                        Eigen::Matrix3d* out_C_ab,
                        Eigen::Matrix3d* out_J_ab) noexcept;

            // -----------------------------------------------------------------------------
            // SO(3) Logarithmic Map (Rotation to Vector)
            // -----------------------------------------------------------------------------

            /**
             * @brief Computes the axis-angle vector from a 3x3 rotation matrix using the logarithmic map.
             * @param C_ab 3x3 rotation matrix.
             * @param eps Numerical stability threshold (default = 1e-6).
             * @return 3x1 axis-angle vector, or zero vector if no valid axis is found.
             */
            Eigen::Vector3d rot2vec(const Eigen::Ref<const Eigen::Matrix3d>& C_ab, double eps = 1e-6) noexcept;

            // -----------------------------------------------------------------------------
            // SO(3) Left Jacobian
            // -----------------------------------------------------------------------------

            /**
             * @brief Computes the 3x3 left Jacobian of SO(3) from an axis-angle vector.
             * @param aaxis_ba 3x1 axis-angle vector.
             * @param numTerms Number of terms in the Taylor series (0 = analytical, >0 = numerical).
             * @return 3x3 Jacobian matrix.
             */
            Eigen::Matrix3d vec2jac(const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba, unsigned int numTerms = 0) noexcept;

            // -----------------------------------------------------------------------------
            // SO(3) Inverse Left Jacobian
            // -----------------------------------------------------------------------------

            /**
             * @brief Computes the 3x3 inverse left Jacobian of SO(3) from an axis-angle vector.
             * @param aaxis_ba 3x1 axis-angle vector.
             * @param numTerms Number of terms in the Taylor series (0 = analytical, >0 = numerical, max 20).
             * @return 3x3 inverse Jacobian matrix, or identity if numTerms > 20.
             */
            Eigen::Matrix3d vec2jacinv(const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba, unsigned int numTerms = 0) noexcept;

        }  // namespace so3
    }  // namespace liemath
}  // namespace slam