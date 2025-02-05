#pragma once

#include <Eigen/Core>

/// Special Orthogonal Group (SO3) - Lie Group Mathematics
namespace slam {
    namespace so3 {

        // -----------------------------------------------------------------------------
        /**
         * @brief Represents a 3D axis-angle rotation vector.
         * 
         * The magnitude of the vector represents the angle of rotation (in radians),
         * while its direction represents the rotation axis. The axis is normalized
         * to ensure a unit-length representation.
         * 
         * This follows the right-hand rule: counterclockwise rotation from frame 'a' to 'b'.
         */
        using AxisAngle = Eigen::Vector3d;

        // -----------------------------------------------------------------------------
        /**
         * @brief Represents a 3x3 rotation matrix.
         * 
         * This matrix transforms points from frame 'a' to frame 'b', following the convention:
         * 
         *    p_b = C_ba * p_a
         * 
         * where C_ba is the rotation matrix from frame 'a' to frame 'b'.
         */
        using RotationMatrix = Eigen::Matrix3d;

    }  // namespace so3
}  // namespace slam
