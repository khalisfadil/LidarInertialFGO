#pragma once

#include <Eigen/Dense>
#include "Utils/colorMode.hpp"

namespace slam {

    struct MapConfig {

        double resolution = 1.0;           ///< The resolution of the map in meters per voxel.
        double mapMaxDistance = 500.0;   ///< The maximum distance that can be reached in the map.
        Eigen::Vector3d mapOrigin = Eigen::Vector3d::Zero();
        unsigned int maxPointsPerVoxel = 20;
        ColorMode colorMode = ColorMode::Occupancy;

        double tolerance = 0.1;
        size_t min_size = 3;
        size_t max_size = 30;
        size_t max_frames = 5;

    };

    struct ProcessConfig {

        double mapMaxDistance = 300.0;   ///< The maximum distance that can be reached in the map.
        double mapMinDistance = 15.0;   ///< The maximum distance that can be reached in the map.
        Eigen::Vector3d mapOrigin = Eigen::Vector3d::Zero();
    };

} // namespace slam
