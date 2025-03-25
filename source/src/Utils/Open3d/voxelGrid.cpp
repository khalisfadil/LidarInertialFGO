#include "Utils/Open3d/voxelGrid.hpp"

namespace slam {

    std::shared_ptr<open3d::geometry::VoxelGrid> createVoxelGrid(
        const std::vector<Voxel3D>& voxels, const Eigen::Vector3d& origin, double resolution) {
        auto voxel_grid = std::make_shared<open3d::geometry::VoxelGrid>();
        voxel_grid->voxel_size_ = resolution;
        voxel_grid->origin_ = origin;

        std::cout << "[createVoxelGrid] Origin: " << voxel_grid->origin_.transpose() << "\n";
        std::cout << "[createVoxelGrid] Voxel size: " << voxel_grid->voxel_size_ << "\n";
        std::cout << "[createVoxelGrid] Input voxels: " << voxels.size() << "\n";

        tbb::concurrent_vector<std::pair<Eigen::Vector3i, open3d::geometry::Voxel>> temp_voxels;
        temp_voxels.reserve(voxels.size());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, voxels.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    const auto& voxel = voxels[i];
                    Eigen::Vector3i grid_index(voxel.key.x, voxel.key.y, voxel.key.z);
                    Eigen::Vector3d color(
                        static_cast<double>(voxel.color.x()) / 255.0,
                        static_cast<double>(voxel.color.y()) / 255.0,
                        static_cast<double>(voxel.color.z()) / 255.0
                    );
                    // Debug first voxel
                    if (i == 0) {
                        Eigen::Vector3d world_pos = voxel.computeGridToWorld(origin, resolution); // Use Voxel3D method
                        std::cout << "[createVoxelGrid] First voxel grid index: " << grid_index.x() << " "
                                << grid_index.y() << " " << grid_index.z() << "\n";
                        std::cout << "[createVoxelGrid] First voxel world position: " << world_pos.transpose() << "\n";
                        std::cout << "[createVoxelGrid] First voxel raw color: " << voxel.color.x() << " "
                                << voxel.color.y() << " " << voxel.color.z() << "\n";
                        std::cout << "[createVoxelGrid] First voxel normalized color: " << color.transpose() << "\n";
                    }
                    temp_voxels.emplace_back(grid_index, open3d::geometry::Voxel(grid_index, color));
                }
            });

        std::cout << "[createVoxelGrid] Temp voxels created: " << temp_voxels.size() << "\n";

        voxel_grid->voxels_.reserve(temp_voxels.size());
        for (const auto& [grid_index, voxel] : temp_voxels) {
            voxel_grid->voxels_.emplace(grid_index, voxel);
        }

        std::cout << "[createVoxelGrid] Voxels assigned: " << voxel_grid->voxels_.size() << "\n";

        return voxel_grid;
    }

    std::shared_ptr<open3d::geometry::TriangleMesh> createVehicleMesh(
        const Eigen::Vector3d& NED, const Eigen::Vector3d& RPY) {
        auto vehicle_mesh = std::make_shared<open3d::geometry::TriangleMesh>();
        std::vector<Eigen::Vector3d> local_vertices = {
            {10.0, 0.0, 0.0},    // Front tip (10 meters long)
            {-10.0, -5.0, 0.0},  // Rear left (10 meters wide)
            {-10.0, 5.0, 0.0}    // Rear right
        };

        Eigen::AngleAxisd rollAngle(RPY.x(), Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(RPY.y(), Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(RPY.z(), Eigen::Vector3d::UnitZ());
        Eigen::Matrix3d R = Eigen::Matrix3d(yawAngle * pitchAngle * rollAngle);
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,3>(0,0) = R;
        T.block<3,1>(0,3) = NED;

        std::vector<Eigen::Vector3d> world_vertices(local_vertices.size());
        for (size_t i = 0; i < local_vertices.size(); ++i) {
            world_vertices[i] = T.block<3,3>(0,0) * local_vertices[i] + T.block<3,1>(0,3);
        }

        vehicle_mesh->vertices_ = world_vertices;
        // Define the single triangular face: connects front (0) to rear left (1) to rear right (2)
        vehicle_mesh->triangles_.push_back(Eigen::Vector3i(0, 1, 2));
        vehicle_mesh->vertex_colors_ = {
            {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0}
        };

        return vehicle_mesh;
    }

} // namespace slam