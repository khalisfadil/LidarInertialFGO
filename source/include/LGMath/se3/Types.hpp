#pragma once

#include <Eigen/Core>

/// Special Euclidean Group SE(3) - Lie Group Math
namespace slam {
    namespace liemath {
        namespace se3 {
            
            // -----------------------------------------------------------------------------
            /**
             * \brief A 3D translation vector.
             * \details Represents the displacement from frame `a` to frame `b`, 
             * expressed in frame `a`.
             *
             *   r_ba_ina = translation from `a` to `b`, expressed in `a`
             */
            using TranslationVector = Eigen::Vector3d;

            // -----------------------------------------------------------------------------
            /**
             * \brief A Lie algebra vector representing an element of se(3).
             * \details This vector is composed of a stacked translation and axis-angle 
             * rotation representation.
             *
             *   xi_ba = [  rho_ba  ]  // Translation component
             *           [ aaxis_ba ]  // Axis-angle rotation component
             *
             * where:
             * - `rho_ba` is the translational part.
             * - `aaxis_ba` is the axis-angle rotational part.
             */
            using LieAlgebra = Eigen::Matrix<double, 6, 1>;

            // -----------------------------------------------------------------------------
            /**
             * \brief Covariance matrix associated with a Lie algebra vector.
             * \details This 6x6 matrix represents the uncertainty in both the translational
             * and rotational components of a Lie algebra vector.
             */
            using LieAlgebraCovariance = Eigen::Matrix<double, 6, 6>;

            // -----------------------------------------------------------------------------
            /**
             * \brief A 4x4 transformation matrix representing an SE(3) element.
             * \details Used to transform points from frame `a` to frame `b`. 
             * Expressed as:
             *
             *   T_ba = [ C_ba  -C_ba * r_ba_ina ]
             *          [  0        1          ]
             *
             * where:
             * - `C_ba` is the rotation matrix (3x3).
             * - `r_ba_ina` is the translation vector (3x1).
             */
            using TransformationMatrix = Eigen::Matrix4d;

        }  // namespace se3
    } // liemath
}  // namespace slam
