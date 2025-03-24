#pragma once

#include <Eigen/Dense>
#include <vector>
#include <limits>

#include "Utils/point.hpp"
#include "Utils/cellKey.hpp"

namespace slam {

    struct Voxel3D {
        unsigned int frameID = std::numeric_limits<unsigned int>::max();    // 4 bytes
        CellKey key;                                                        // 12 bytes
        double timestamp = 0.0;                                             // 8 bytes
        Eigen::Vector3d averagePointAtt = Eigen::Vector3d::Zero();          // 24 bytes, avg attributes (e.g., intensity)
        Eigen::Vector3d averagePointPose = Eigen::Vector3d::Zero();         // 24 bytes, avg position
        unsigned int counter = 0;                                           // 4 bytes, point count
        Eigen::Vector3i color = Eigen::Vector3i::Zero();                    // 12 bytes colour coded

        Eigen::Vector3d computeGridToWorld(const Eigen::Vector3d& origin, 
                                           double resolution) const {
            static const Eigen::Vector3d halfResolution(resolution / 2, resolution / 2, resolution / 2);
            Eigen::Vector3d gridIndex(key.x, key.y, key.z);
            return origin + gridIndex * resolution + halfResolution;        // Implicit cast is fine
        }
    };

}  // namespace slam