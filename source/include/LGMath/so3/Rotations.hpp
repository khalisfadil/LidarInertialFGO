#pragma once

#include <Eigen/Dense>
#include <iostream>

namespace slam {
    namespace liemath {
        namespace so3 {
            
            // -----------------------------------------------------------------------------
            /**
             * @brief A lightweight rotation matrix class for efficient SO(3) operations.
             * 
             * This class provides fast and minimal rotation matrix operations, 
             * suitable for real-time robotics and SLAM applications.
             */
            class Rotation {
                public:

                    // -----------------------------------------------------------------------------
                    /** @brief Default constructor (identity rotation). */
                    Rotation();

                    // -----------------------------------------------------------------------------
                    /** @brief Construct from an existing Eigen 3x3 matrix. */
                    explicit Rotation(const Eigen::Matrix3d& C);

                    // -----------------------------------------------------------------------------
                    /** @brief Construct a rotation matrix from an axis-angle vector using the exponential map. */
                    explicit Rotation(const Eigen::Vector3d& aaxis_ab, unsigned int numTerms = 0);

                    // -----------------------------------------------------------------------------
                    /** @brief Construct a rotation matrix from an Eigen vector (must be 3x1). */
                    explicit Rotation(const Eigen::VectorXd& aaxis_ab);

                    // -----------------------------------------------------------------------------
                    /** @brief Retrieve the underlying rotation matrix. */
                    const Eigen::Matrix3d& matrix() const;

                    // -----------------------------------------------------------------------------
                    /** @brief Convert rotation matrix to Lie algebra representation (logarithmic map). */
                    Eigen::Vector3d vec() const;

                    // -----------------------------------------------------------------------------
                    /** @brief Compute the inverse (transpose) of the rotation matrix. */
                    Rotation inverse() const;

                    // -----------------------------------------------------------------------------
                    /** 
                     * @brief Reproject the matrix onto SO(3) if numerical errors accumulate.
                     * @param[in] force If false, only reprojects when numerical instability is detected.
                     */
                    void reproject(bool force = true);

                    // -----------------------------------------------------------------------------
                    /** @brief Multiply this rotation matrix in-place with another rotation. */
                    Rotation& operator*=(const Rotation& C_rhs) noexcept;

                    // -----------------------------------------------------------------------------
                    /** @brief Multiply this rotation matrix with another rotation and return the result. */
                    Rotation operator*(const Rotation& C_rhs) const;

                    // -----------------------------------------------------------------------------
                    /** @brief Divide this rotation matrix in-place by another rotation (multiplication by inverse). */
                    Rotation& operator/=(const Rotation& C_rhs) noexcept;

                    // -----------------------------------------------------------------------------
                    /** @brief Divide this rotation matrix by another rotation and return the result. */
                    Rotation operator/(const Rotation& C_rhs) const;

                    // -----------------------------------------------------------------------------
                    /** @brief Apply rotation to a 3D point vector. */
                    Eigen::Vector3d operator*(const Eigen::Ref<const Eigen::Vector3d>& p_a) const noexcept;

                private:
                
                    // -----------------------------------------------------------------------------
                    Eigen::Matrix3d C_ba_; ///< Rotation matrix representing transformation from frame A to B.
            };
        }  // namespace so3
    } // liemath
}  // namespace slam

// -----------------------------------------------------------------------------
/** @brief Stream output operator for printing the rotation matrix. */
std::ostream& operator<<(std::ostream& out, const slam::liemath::so3::Rotation& T);
