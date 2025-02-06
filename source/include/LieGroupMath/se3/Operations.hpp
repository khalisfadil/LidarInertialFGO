#pragma once

#include <include/lieGroupMath/so3/Operations.hpp>
#include <Eigen/Core>

/// Special Euclidean (SE3) Lie Group Mathematics
namespace slam {
    namespace liemath {
        namespace se3 {

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs the 4x4 skew-symmetric matrix (hat operator) 
             *        from translation and axis-angle vectors.
             * @param rho    3x1 translation vector.
             * @param aaxis  3x1 axis-angle rotation vector.
             * @return       4x4 skew-symmetric matrix.
             */
            Eigen::Matrix4d hat(const Eigen::Ref<const Eigen::Vector3d>& rho, 
                                const Eigen::Ref<const Eigen::Vector3d>& aaxis) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs the 4x4 skew-symmetric matrix (hat operator) 
             *        from a 6x1 se3 algebra vector.
             * @param xi  6x1 se3 algebra vector (translation + axis-angle rotation).
             * @return    4x4 skew-symmetric matrix.
             */
            Eigen::Matrix4d hat(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs the 6x6 curly-hat matrix from translation and axis-angle vectors.
             * @param rho    3x1 translation vector.
             * @param aaxis  3x1 axis-angle rotation vector.
             * @return       6x6 curly-hat matrix.
             */
            Eigen::Matrix<double, 6, 6> curlyhat(const Eigen::Ref<const Eigen::Vector3d>& rho, 
                                                const Eigen::Ref<const Eigen::Vector3d>& aaxis) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Constructs the 6x6 curly-hat matrix from a 6x1 se3 algebra vector.
             * @param xi  6x1 se3 algebra vector (translation + axis-angle rotation).
             * @return    6x6 curly-hat matrix.
             */
            Eigen::Matrix<double, 6, 6> curlyhat(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Converts a 3D point into a 4x6 matrix (circle-dot operator).
             * @param p      3D point.
             * @param scale  Scaling factor (default = 1).
             * @return       4x6 matrix.
             */
            Eigen::Matrix<double, 4, 6> point2fs(const Eigen::Ref<const Eigen::Vector3d>& p, double scale = 1.0) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Converts a 3D point into a 6x4 matrix (double-circle operator).
             * @param p      3D point.
             * @param scale  Scaling factor (default = 1).
             * @return       6x4 matrix.
             */
            Eigen::Matrix<double, 6, 4> point2sf(const Eigen::Ref<const Eigen::Vector3d>& p, double scale = 1.0) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the SE(3) transformation matrix using the analytical exponential map.
             * @param rho_ba       Translation vector.
             * @param aaxis_ba     Axis-angle rotation vector.
             * @param out_C_ab     Output rotation matrix.
             * @param out_r_ba_ina Output translation vector.
             */
            void vec2tran_analytical(const Eigen::Ref<const Eigen::Vector3d>& rho_ba, 
                                    const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba,
                                    Eigen::Matrix3d* out_C_ab, 
                                    Eigen::Vector3d* out_r_ba_ina) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the SE(3) transformation matrix using a numerical approximation.
             * @param rho_ba       Translation vector.
             * @param aaxis_ba     Axis-angle rotation vector.
             * @param out_C_ab     Output rotation matrix.
             * @param out_r_ba_ina Output translation vector.
             * @param numTerms     Number of terms in the approximation (default = 0 for analytical solution).
             */
            void vec2tran_numerical(const Eigen::Ref<const Eigen::Vector3d>& rho_ba, 
                                    const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba,
                                    Eigen::Matrix3d* out_C_ab, 
                                    Eigen::Vector3d* out_r_ba_ina,
                                    unsigned int numTerms = 0) noexcept;
                                    
            // -----------------------------------------------------------------------------
            /**  Computes the SE(3) transformation matrix from a given se3 algebra vector 
             * using either an analytical or numerical approach.
             *
             * This function computes the transformation matrix `T_ba`, decomposing it into
             * a rotation matrix `C_ab` and a translation vector `r_ba_ina`. The computation 
             * method depends on the `numTerms` parameter:
             *
             * - If `numTerms == 0`, an **analytical solution** is used, which is exact and
             *  based on the exponential map.
             * - If `numTerms > 0`, a **numerical approximation** is used, computed via a 
             *  truncated power series expansion.
             *
             * @param[in]  xi_ba        6x1 se3 algebra vector (twist representation).
             *                         The first 3 elements represent translation `rho_ba`,
             *                         and the last 3 elements represent rotation `aaxis_ba`.
             * @param[out] out_C_ab     Pointer to a 3x3 rotation matrix `C_ab`.
             * @param[out] out_r_ba_ina Pointer to a 3x1 translation vector `r_ba_ina`.
             *                         This represents the translation of frame `b` w.r.t. `a`,
             *                         expressed in frame `a`.
             * @param[in]  numTerms     Number of terms for the numerical approximation.
             *                         If `numTerms == 0`, an analytical solution is used.
             *
             * @note `out_C_ab` and `out_r_ba_ina` must not be null pointers.
             */
            void vec2tran(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ba,
                        Eigen::Matrix3d* out_C_ab, Eigen::Vector3d* out_r_ba_ina,
                        unsigned int numTerms) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the SE(3) transformation matrix using the exponential map.
             * @param xi_ba    6x1 se3 algebra vector.
             * @param numTerms Number of terms in the approximation (default = 0 for analytical solution).
             * @return         4x4 transformation matrix.
             */
            Eigen::Matrix4d vec2tran(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ba, 
                                    unsigned int numTerms = 0) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the logarithmic map (inverse of exponential map) for an SE(3) transformation.
             * @param C_ab        Rotation matrix.
             * @param r_ba_ina    Translation vector.
             * @return            6x1 se3 algebra vector.
             */
            Eigen::Matrix<double, 6, 1> tran2vec(const Eigen::Ref<const Eigen::Matrix3d>& C_ab, 
                                                const Eigen::Ref<const Eigen::Vector3d>& r_ba_ina) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the logarithmic map for a 4x4 SE(3) transformation matrix.
             * @param T_ab  4x4 transformation matrix.
             * @return      6x1 se3 algebra vector.
             */
            Eigen::Matrix<double, 6, 1> tran2vec(const Eigen::Ref<const Eigen::Matrix4d>& T_ab) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the 6x6 adjoint transformation matrix.
             * @param C_ab        Rotation matrix.
             * @param r_ba_ina    Translation vector.
             * @return            6x6 adjoint transformation matrix.
             */
            Eigen::Matrix<double, 6, 6> tranAd(const Eigen::Ref<const Eigen::Matrix3d>& C_ab, 
                                            const Eigen::Ref<const Eigen::Vector3d>& r_ba_ina) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the 6x6 adjoint transformation matrix from a 4x4 transformation matrix.
             * @param T_ab  4x4 transformation matrix.
             * @return      6x6 adjoint transformation matrix.
             */
            Eigen::Matrix<double, 6, 6> tranAd(const Eigen::Ref<const Eigen::Matrix4d>& T_ab) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the Q matrix for SE(3).
             * @param rho_ba   Translation vector.
             * @param aaxis_ba Axis-angle rotation vector.
             * @return         3x3 Q matrix.
             */
            Eigen::Matrix3d vec2Q(const Eigen::Ref<const Eigen::Vector3d>& rho_ba, 
                                const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the Q matrix for SE(3) from a 6x1 se3 algebra vector.
             * @param xi_ba  6x1 se3 algebra vector.
             * @return       3x3 Q matrix.
             */
            Eigen::Matrix3d vec2Q(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ba) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the 6x6 left Jacobian of SE(3).
             * @param rho_ba   Translation vector.
             * @param aaxis_ba Axis-angle rotation vector.
             * @return         6x6 left Jacobian matrix.
             */
            Eigen::Matrix<double, 6, 6> vec2jac(const Eigen::Ref<const Eigen::Vector3d>& rho_ba, 
                                                const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the 6x6 left Jacobian of SE(3) from a 6x1 se3 algebra vector.
             * @param xi_ba    6x1 se3 algebra vector.
             * @param numTerms Number of terms in the approximation (default = 0 for analytical solution).
             * @return         6x6 left Jacobian matrix.
             */
            Eigen::Matrix<double, 6, 6> vec2jac(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ba, 
                                                unsigned int numTerms = 0) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the inverse of the left Jacobian for SE(3).
             * @param rho_ba   Translation vector.
             * @param aaxis_ba Axis-angle rotation vector.
             * @return         6x6 inverse left Jacobian matrix.
             */
            Eigen::Matrix<double, 6, 6> vec2jacinv(const Eigen::Ref<const Eigen::Vector3d>& rho_ba, 
                                                const Eigen::Ref<const Eigen::Vector3d>& aaxis_ba) noexcept;

            // -----------------------------------------------------------------------------
            /**
             * @brief Computes the inverse of the left Jacobian for SE(3) using a numerical approximation.
             * @param xi_ba    6x1 se3 algebra vector.
             * @param numTerms Number of terms in the approximation (default = 0 for analytical solution).
             * @return         6x6 inverse left Jacobian matrix.
             */
            Eigen::Matrix<double, 6, 6> vec2jacinv(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ba, 
                                                unsigned int numTerms = 0) noexcept;

        }  // namespace se3
    } // liemath
}  // namespace slam
