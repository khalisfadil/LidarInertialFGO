#pragma once

#include <Eigen/Dense>
#include <vector>

namespace slam {

    /**
     * @brief 3D point structure with spatial coordinates, attributes, and timing information.
     */
    struct Point3D {
        Eigen::Vector3d Pt = Eigen::Vector3d::Zero();                  // 3D coordinates (x, y, z)
        Eigen::Vector3d Att = Eigen::Vector3d::Zero();                 // Attributes (e.g., intensity, reflectivity, NIR)
        double Timestamp = 0.0;                                         // Per-point timestamps
    };

}  // namespace slam