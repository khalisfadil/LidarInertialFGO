#pragma once

#include <Eigen/Dense>
#include <vector>

#include "Utils/point.hpp"

namespace slam {

    /**
     * @brief Lightweight data frame structure storing a Point3D vector for a specific frame.
     *        Represents a full point cloud with associated metadata.
     */
    struct ClusterExtractorDataFrame {
        unsigned int frameID = std::numeric_limits<unsigned int>::max();                                           // Frame identifier
        double timestamp = 0.0;                                     // Frame timestamp
        std::vector<Point3D> pointcloud;                            // Full point cloud for the frame
    };

    /**
     * @brief Data frame structure for occupancy mapping, containing a point cloud with attributes and metadata.
     */
    struct OccupancyMapDataFrame {
        unsigned int frameID = std::numeric_limits<unsigned int>::max(); // Frame identifier
        double timestamp = 0.0;                                          // Frame timestamp
        Eigen::Vector3d vehiclePosition = Eigen::Vector3d::Zero();       // Vehicle position for this frame
        std::vector<Point3D> pointcloud;                                 // Point cloud with position and attributes

    };

    struct VehiclePoseDataFrame {
        unsigned int frameID = std::numeric_limits<unsigned int>::max(); // Frame identifier
        double timestamp = 0.0;                                          // Frame timestamp
        Eigen::Vector3d NED = Eigen::Vector3d::Zero();       // Vehicle position for this frame
        Eigen::Vector3d RPY = Eigen::Vector3d::Zero();       // Vehicle position for this frame
    };

}  // namespace slam