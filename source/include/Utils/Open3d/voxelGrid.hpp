#pragma once

#include <algorithm> 

#include <open3d/Open3D.h>

#include "Utils/voxel.hpp"

#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>

namespace slam {

    // Creates a VoxelGrid from a vector of Voxel3D objects
    std::shared_ptr<open3d::geometry::VoxelGrid> createVoxelGrid(
        const std::vector<Voxel3D>& voxels,
        const Eigen::Vector3d& origin,
        double resolution);

    std::shared_ptr<open3d::geometry::TriangleMesh> createVehicleMesh(
        const Eigen::Vector3d& NED, const Eigen::Vector3d& RPY);

} // namespace slam