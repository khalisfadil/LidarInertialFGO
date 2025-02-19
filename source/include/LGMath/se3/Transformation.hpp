#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

namespace slam {
    namespace liemath {
        namespace se3 {
        
        // -----------------------------------------------------------------------------
        /**
         * \class Transformation
         * \brief Represents a rigid body transformation in SE(3).
         * 
         * This class models a rigid body transformation in the Special Euclidean group SE(3), 
         * consisting of a **rotation matrix (C_ba_)** and a **translation vector (r_ab_inb_)**.
         *
         * - **C_ba_ (rotation matrix)**: Represents the rotation from frame **a** to frame **b**.
         * - **r_ab_inb_ (translation vector)**: Represents the position of frame **a** in frame **b**, expressed in frame **b**.
         * - **r_ba_ina (not stored directly)**: Would be the inverse translation (position of frame **b** in frame **a**).
         */
        class Transformation {

            public:

                // -----------------------------------------------------------------------------
                /** \brief Default constructor (identity transformation) */
                Transformation();

                // -----------------------------------------------------------------------------
                /** \brief Copy constructor. */
                Transformation(const Transformation&) = default;

                // -----------------------------------------------------------------------------
                /** \brief Move constructor. */
                Transformation(Transformation&&) = default;

                // -----------------------------------------------------------------------------
                /** \brief Construct transformation from a 4x4 homogeneous transformation matrix */
                explicit Transformation(const Eigen::Ref<const Eigen::Matrix4d>& T);

                // -----------------------------------------------------------------------------
                /**
                 * \brief Construct transformation from rotation and translation components.
                 * 
                 * \param[in] C_ba Rotation matrix from frame **a** to **b**.
                 * \param[in] r_ba_ina Translation vector (position of frame **b** in frame **a**, expressed in frame **a**).
                 */
                explicit Transformation(const Eigen::Ref<const Eigen::Matrix3d>& C_ba,
                                        const Eigen::Ref<const Eigen::Vector3d>& r_ba_ina);

                // -----------------------------------------------------------------------------
                /**
                 * \brief Construct transformation from a 6D vector using the exponential map.
                 * 
                 * \param[in] xi_ab 6x1 se(3) algebra vector (translation + axis-angle rotation).
                 * \param[in] numTerms Number of terms in the series expansion (0 for analytical solution).
                 */
                explicit Transformation(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ab, unsigned int numTerms = 0);

                // -----------------------------------------------------------------------------
                /**
                 * \brief Construct transformation from a general Eigen vector.
                 * 
                 * The input vector **must be 6x1**, otherwise an exception is thrown.
                 * \param[in] xi_ab 6x1 se(3) algebra vector.
                 */
                explicit Transformation(const Eigen::Ref<const Eigen::VectorXd>& xi_ab);

                // -----------------------------------------------------------------------------
                /** \brief Destructor (default implementation). */
                virtual ~Transformation() = default;

                // -----------------------------------------------------------------------------
                /** \brief Copy assignment operator. */
                virtual Transformation& operator=(const Transformation&) = default;

                // -----------------------------------------------------------------------------
                /** \brief Move assignment operator. */
                virtual Transformation& operator=(Transformation&&) = default;

                // -----------------------------------------------------------------------------
                /** \brief Gets the 4x4 homogeneous transformation matrix representation. */
                Eigen::Matrix4d matrix() const noexcept;

                // -----------------------------------------------------------------------------
                /** 
                 * \brief Gets the underlying rotation matrix (C_ba).
                 * 
                 * **C_ba_ (rotation matrix):** Rotation from frame **a** to frame **b**.
                 * \return 3x3 rotation matrix.
                 */
                const Eigen::Matrix3d& C_ba() const noexcept;

                // -----------------------------------------------------------------------------
                /**
                 * \brief Gets the translation vector r_ab_inb.
                 * 
                 * **r_ab_inb_ (translation vector):** Position of frame **a** in frame **b**, expressed in frame **b**.
                 * \return 3x1 translation vector.
                 */
                const Eigen::Vector3d& r_ab_inb() const noexcept;

                // -----------------------------------------------------------------------------
                /**
                 * \brief Computes r_ba_ina = -C_ba.transpose() * r_ab_inb.
                 * 
                 * This function computes the inverse translation, which is the **position of frame b in frame a**,
                 * expressed in frame **a**.
                 * \return 3x1 inverse translation vector.
                 */
                Eigen::Vector3d r_ba_ina() const noexcept;

                // -----------------------------------------------------------------------------
                /** \brief Compute the logarithmic map (inverse of the exponential map). */
                Eigen::Matrix<double, 6, 1> vec() const noexcept;

                // -----------------------------------------------------------------------------
                /** \brief Compute the inverse of the transformation. */
                Transformation inverse() const noexcept;

                // -----------------------------------------------------------------------------
                /** \brief Compute the 6x6 adjoint transformation matrix. */
                Eigen::Matrix<double, 6, 6> adjoint() const noexcept;

                // -----------------------------------------------------------------------------
                /**
                 * \brief Reprojects the transformation matrix onto SE(3).
                 * 
                 * \param[in] force If true, forces reprojection. If false, only reprojects if the determinant of **C_ba** is far from 1.
                 */
                void reproject(bool force = true) noexcept;

                // -----------------------------------------------------------------------------
                /** \brief In-place right-hand side multiplication with another transformation. */
                virtual Transformation& operator*=(const Transformation& T_rhs) noexcept;

                // -----------------------------------------------------------------------------
                /** \brief Right-hand side multiplication with another transformation. */
                virtual Transformation operator*(const Transformation& T_rhs) const noexcept;

                // -----------------------------------------------------------------------------
                /** \brief In-place right-hand side multiplication with the inverse of another transformation. */
                virtual Transformation& operator/=(const Transformation& T_rhs) noexcept;

                // -----------------------------------------------------------------------------
                /** \brief Right-hand side multiplication with the inverse of another transformation. */
                virtual Transformation operator/(const Transformation& T_rhs) const noexcept;

                // -----------------------------------------------------------------------------
                /** \brief Right-hand side multiplication with a homogeneous vector (4D point). */
                Eigen::Vector4d operator*(const Eigen::Ref<const Eigen::Vector4d>& p_a) const noexcept;

            private:

                // -----------------------------------------------------------------------------
                /** \brief Rotation matrix from frame a to frame b */
                Eigen::Matrix3d C_ba_;

                // -----------------------------------------------------------------------------
                /** \brief Translation vector (position of frame a in frame b, expressed in frame b) */
                Eigen::Vector3d r_ab_inb_;
            };

    }  // namespace se3
    } // liemath
}  // namespace slam

// -----------------------------------------------------------------------------
/** \brief Print transformation */
std::ostream& operator<<(std::ostream& out, const slam::liemath::se3::Transformation& T);
