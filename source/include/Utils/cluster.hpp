#pragma once

#include <Eigen/Dense>
#include <vector>
#include <limits>  // Added for std::numeric_limits

namespace slam {

    /**
     * @brief Lightweight cluster structure storing indices of points from a Point3D vector.
     *        Does not duplicate point data, only references it by index.
     */
    struct Cluster3D {
        unsigned int clusterID = std::numeric_limits<unsigned int>::max();  // Unique identifier for the cluster
        unsigned int frameID = std::numeric_limits<unsigned int>::max();    // Frame identifier
        double timestamp = 0.0;                                             // Frame timestamp (in seconds)
        std::vector<size_t> idx;                                            // Indices of points in the cluster
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();                 // Centroid of the cluster (x, y, z)
        Eigen::Vector3d averageAtt = Eigen::Vector3d::Zero();               // Average attribute (e.g., color, intensity)
        Eigen::Vector3d minBound = Eigen::Vector3d::Zero();                 // Minimum corner of bounding box (x, y, z)
        Eigen::Vector3d maxBound = Eigen::Vector3d::Zero();                 // Maximum corner of bounding box (x, y, z)
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();                 // Velocity of the cluster (m/s)
        bool isDynamic = false;                                             // Classification: true if dynamic, false if static

    };

}  // namespace slam