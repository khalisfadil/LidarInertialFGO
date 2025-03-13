// Done
#pragma once

#include <Eigen/Core>
#include "LGMath/so3/Operations.hpp"

namespace slam {
    namespace liemath {
        namespace se3 {

        // -----------------------------------------------------------------------------
        // SE(3) Hat Operators
        // -----------------------------------------------------------------------------

        /**
         * @brief Constructs the 4x4 skew-symmetric matrix (hat operator) from translation and axis-angle vectors.
         * @param rho    3x1 translation vector.
         * @param aaxis  3x1 axis-angle rotation vector.
         * @return       4x4 skew-symmetric matrix.
         */
        Eigen::Matrix4d hat(const Eigen::Ref<const Eigen::Vector3d>& rho,
                            const Eigen::Ref<const Eigen::Vector3d>& aaxis) noexcept;

        /**
         * @brief Constructs the 4x4 skew-symmetric matrix (hat operator) from a 6x1 SE(3) algebra vector.
         * @param xi  6x1 SE(3) algebra vector (translation + axis-angle rotation).
         * @return    4x4 skew-symmetric matrix.
         */
        Eigen::Matrix4d hat(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi) noexcept;

        // -----------------------------------------------------------------------------
        // SE(3) Curly Hat Operators
        // -----------------------------------------------------------------------------

        /**
         * @brief Constructs the 6x6 curly-hat matrix from translation and axis-angle vectors.
         * @param rho    3x1 translation vector.
         * @param aaxis  3x1 axis-angle rotation vector.
         * @return       6x6 curly-hat matrix.
         */
        Eigen::Matrix<double, 6, 6> curlyhat(const Eigen::Ref<const Eigen::Vector3d>& rho,
                                            const Eigen::Ref<const Eigen::Vector3d>& aaxis) noexcept;

        /**
         * @brief Constructs the 6x6 curly-hat matrix from a 6x1 SE(3) algebra vector.
         * @param xi  6x1 SE(3) algebra vector (translation + axis-angle rotation).
         * @return    6x6 curly-hat matrix.
         */
        Eigen::Matrix<double, 6, 6> curlyhat(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi) noexcept;

        // -----------------------------------------------------------------------------
        // Point Transformations to Lie Algebra
        // -----------------------------------------------------------------------------

        /**
         * @brief Converts a 3D point into a 4x6 matrix (circle-dot operator).
         * @param p      3D point.
         * @param scale  Scaling factor (default = 1.0).
         * @return       4x6 matrix.
         */
        Eigen::Matrix<double, 4, 6> point2fs(const Eigen::Ref<const Eigen::Vector3d>& p, double scale = 1.0) noexcept;

        /**
         * @brief Converts a 3D point into a 6x4 matrix (double-circle operator).
         * @param p      3D point.
         * @param scale  Scaling factor (default = 1.0).
         * @return       6x4 matrix.
         */
        Eigen::Matrix<double, 6, 4> point2sf(const Eigen::Ref<const Eigen::Vector3d>& p, double scale = 1.0) noexcept;

        // -----------------------------------------------------------------------------
        // SE(3) Exponential Map (Vector to Transformation)
        // -----------------------------------------------------------------------------

        /**
         * @brief Computes the SE(3) transformation analytically from translation and axis-angle vectors.
         * @param rho_ba       3x1 translation vector.
         * @param aaxis_ba     3x1 axis-angle rotation vector.
         * @param out_C_ab     Output pointer to 3x3 rotation matrix (must not be null).
         * @param out_r_ba_ina Output pointer to 3x1 translation vector (must not be null).
         * @pre                out_C_ab and out_r_ba_ina must point to valid memory.
         */
        void vec2tran_analytical(const Eigen::Ref<const Eigen::Vector3d>& rho_ba,
                                const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba,
                                Eigen::Matrix3d* out_C_ab,
                                Eigen::Vector3d* out_r_ba_ina) noexcept;

        /**
         * @brief Computes the SE(3) transformation numerically from translation and axis-angle vectors.
         * @param rho_ba       3x1 translation vector.
         * @param aaxis_ba     3x1 axis-angle rotation vector.
         * @param out_C_ab     Output pointer to 3x3 rotation matrix (must not be null).
         * @param out_r_ba_ina Output pointer to 3x1 translation vector (must not be null).
         * @param numTerms     Number of terms in the series approximation.
         * @pre                out_C_ab and out_r_ba_ina must point to valid memory.
         */
        void vec2tran_numerical(const Eigen::Ref<const Eigen::Vector3d>& rho_ba,
                                const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba,
                                Eigen::Matrix3d* out_C_ab,
                                Eigen::Vector3d* out_r_ba_ina,
                                unsigned int numTerms) noexcept;

        /**
         * @brief Computes the SE(3) transformation from a 6x1 SE(3) algebra vector.
         * @param xi_ba        6x1 SE(3) algebra vector (translation + rotation).
         * @param out_C_ab     Output pointer to 3x3 rotation matrix (must not be null).
         * @param out_r_ba_ina Output pointer to 3x1 translation vector (must not be null).
         * @param numTerms     Number of terms (0 = analytical, >0 = numerical).
         * @pre                out_C_ab and out_r_ba_ina must point to valid memory.
         */
        void vec2tran(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ba,
                    Eigen::Matrix3d* out_C_ab,
                    Eigen::Vector3d* out_r_ba_ina,
                    unsigned int numTerms = 0) noexcept;

        /**
         * @brief Computes the 4x4 SE(3) transformation matrix from a 6x1 SE(3) algebra vector.
         * @param xi_ba    6x1 SE(3) algebra vector.
         * @param numTerms Number of terms (0 = analytical, >0 = numerical).
         * @return         4x4 transformation matrix.
         */
        Eigen::Matrix4d vec2tran(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ba,
                                unsigned int numTerms = 0) noexcept;

        // -----------------------------------------------------------------------------
        // SE(3) Logarithmic Map (Transformation to Vector)
        // -----------------------------------------------------------------------------

        /**
         * @brief Computes the logarithmic map from rotation and translation to a 6x1 SE(3) algebra vector.
         * @param C_ab        3x3 rotation matrix.
         * @param r_ba_ina    3x1 translation vector.
         * @return            6x1 SE(3) algebra vector.
         * @note              Uses scalar assignment to avoid vectorization issues with 3x1 vectors.
         */
        Eigen::Matrix<double, 6, 1> tran2vec(const Eigen::Ref<const Eigen::Matrix3d>& C_ab,
                                            const Eigen::Ref<const Eigen::Vector3d>& r_ba_ina) noexcept;

        /**
         * @brief Computes the logarithmic map from a 4x4 SE(3) transformation matrix.
         * @param T_ab  4x4 transformation matrix.
         * @return      6x1 SE(3) algebra vector.
         * @note        Uses scalar assignment to avoid vectorization issues with 3x1 vectors.
         */
        Eigen::Matrix<double, 6, 1> tran2vec(const Eigen::Ref<const Eigen::Matrix4d>& T_ab) noexcept;

        // -----------------------------------------------------------------------------
        // SE(3) Adjoint Transformation
        // -----------------------------------------------------------------------------

        /**
         * @brief Computes the 6x6 adjoint transformation matrix from rotation and translation.
         * @param C_ab        3x3 rotation matrix.
         * @param r_ba_ina    3x1 translation vector.
         * @return            6x6 adjoint matrix.
         */
        Eigen::Matrix<double, 6, 6> tranAd(const Eigen::Ref<const Eigen::Matrix3d>& C_ab,
                                        const Eigen::Ref<const Eigen::Vector3d>& r_ba_ina) noexcept;

        /**
         * @brief Computes the 6x6 adjoint transformation matrix from a 4x4 transformation matrix.
         * @param T_ab  4x4 transformation matrix.
         * @return      6x6 adjoint matrix.
         */
        Eigen::Matrix<double, 6, 6> tranAd(const Eigen::Ref<const Eigen::Matrix4d>& T_ab) noexcept;

        // -----------------------------------------------------------------------------
        // SE(3) Q Matrix
        // -----------------------------------------------------------------------------

        /**
         * @brief Computes the 3x3 Q matrix for SE(3) from translation and axis-angle vectors.
         * @param rho_ba   3x1 translation vector.
         * @param aaxis_ba 3x1 axis-angle rotation vector.
         * @return         3x3 Q matrix.
         */
        Eigen::Matrix3d vec2Q(const Eigen::Ref<const Eigen::Vector3d>& rho_ba,
                            const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba) noexcept;

        /**
         * @brief Computes the 3x3 Q matrix for SE(3) from a 6x1 SE(3) algebra vector.
         * @param xi_ba  6x1 SE(3) algebra vector.
         * @return       3x3 Q matrix.
         */
        Eigen::Matrix3d vec2Q(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ba) noexcept;

        // -----------------------------------------------------------------------------
        // SE(3) Left Jacobian
        // -----------------------------------------------------------------------------

        /**
         * @brief Computes the 6x6 left Jacobian of SE(3) from translation and axis-angle vectors.
         * @param rho_ba   3x1 translation vector.
         * @param aaxis_ba 3x1 axis-angle rotation vector.
         * @return         6x6 left Jacobian matrix.
         */
        Eigen::Matrix<double, 6, 6> vec2jac(const Eigen::Ref<const Eigen::Vector3d>& rho_ba,
                                            const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba) noexcept;

        /**
         * @brief Computes the 6x6 left Jacobian of SE(3) from a 6x1 SE(3) algebra vector.
         * @param xi_ba    6x1 SE(3) algebra vector.
         * @param numTerms Number of terms (0 = analytical, >0 = numerical).
         * @return         6x6 left Jacobian matrix.
         */
        Eigen::Matrix<double, 6, 6> vec2jac(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ba,
                                            unsigned int numTerms = 0) noexcept;

        // -----------------------------------------------------------------------------
        // SE(3) Inverse Left Jacobian
        // -----------------------------------------------------------------------------

        /**
         * @brief Computes the 6x6 inverse left Jacobian of SE(3) from translation and axis-angle vectors.
         * @param rho_ba   3x1 translation vector.
         * @param aaxis_ba 3x1 axis-angle rotation vector.
         * @return         6x6 inverse left Jacobian matrix.
         */
        Eigen::Matrix<double, 6, 6> vec2jacinv(const Eigen::Ref<const Eigen::Vector3d>& rho_ba,
                                            const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba) noexcept;

        /**
         * @brief Computes the 6x6 inverse left Jacobian of SE(3) from a 6x1 SE(3) algebra vector.
         * @param xi_ba    6x1 SE(3) algebra vector.
         * @param numTerms Number of terms (0 = analytical, >0 = numerical, max 20).
         * @return         6x6 inverse left Jacobian matrix.
         * @note           For numTerms > 20, returns identity matrix with a warning (no exception thrown).
         */
        Eigen::Matrix<double, 6, 6> vec2jacinv(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ba,
                                            unsigned int numTerms = 0) noexcept;

        }  // namespace se3
    }  // namespace liemath
}  // namespace slam