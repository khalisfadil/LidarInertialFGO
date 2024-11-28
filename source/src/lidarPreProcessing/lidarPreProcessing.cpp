#include "lidarPreProcessing.hpp"



tsl::robin_map<Eigen::Vector3i, lidarPreProcessing::VoxelData, 
                lidarPreProcessing::Vector3iHash, 
                lidarPreProcessing::Vector3iEqual> 
lidarPreProcessing::constructHashMap(std::vector<Eigen::Vector3f> points_,
                                     std::vector<Eigen::Vector3f> attributes_,
                                     float res)
{   
    tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> insertedVoxels_;
    // Use TBB `parallel_reduce` to parallelize voxel insertion
    auto localMap = tbb::parallel_reduce(
        tbb::blocked_range<uint64_t>(0, points_.size(), 1024),
        tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual>(),
        [&](const tbb::blocked_range<uint64_t>& range,
            tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> localMap) {

            for (uint64_t i = range.begin(); i < range.end(); ++i) {
                const Eigen::Vector3f& position = points_[i];
                const Eigen::Vector3f& attribute = attributes_[i];
                Eigen::Vector3i gridIndex = posToGridIndex(position, res);

                // Access the voxel corresponding to the grid index
                auto& voxel = localMap[gridIndex];

                if (voxel.datas.empty()) {
                    voxel.centerPosition = gridToWorld(gridIndex, res);
                }

                // Explicitly construct PointData and push it to the voxel
                PointData pointData;
                pointData.position = position;
                pointData.attributes = attribute;
                voxel.datas.push_back(std::move(pointData));
            }
            return localMap;
        },
        // Combine local voxel maps into a single map
        [](auto a, auto b) {
            for (auto& [gridIndex, localVoxel] : b) {
                auto& voxel = a[gridIndex];
                if (voxel.datas.empty()) {
                    voxel = std::move(localVoxel);
                } else {
                    voxel.datas.insert(voxel.datas.end(),
                                       std::make_move_iterator(localVoxel.datas.begin()),
                                       std::make_move_iterator(localVoxel.datas.end()));
                }
            }
            return a;
        });

    // Merge `localMap` into `insertedVoxels_`
    for (auto& [gridIndex, localVoxel] : localMap) {
        auto& voxel = insertedVoxels_[gridIndex];
        if (voxel.datas.empty()) {
            voxel = std::move(localVoxel);
        } else {
            voxel.datas.insert(voxel.datas.end(),
                               std::make_move_iterator(localVoxel.datas.begin()),
                               std::make_move_iterator(localVoxel.datas.end()));
        }
    }

    return insertedVoxels_;
}
//##############################################################################
// Convert position to voxel grid index
Eigen::Vector3i lidarPreProcessing::posToGridIndex(const Eigen::Vector3f& pos, const float& res) const {
    Eigen::Array3f scaledPos = (pos).array() * (1.0 / res);
    return Eigen::Vector3i(std::floor(scaledPos.x()), std::floor(scaledPos.y()), std::floor(scaledPos.z()));
}
//##############################################################################
// Convert grid index back to world position (center of voxel)
Eigen::Vector3f lidarPreProcessing::gridToWorld(const Eigen::Vector3i& gridIndex,const float& res) const {
    static const Eigen::Vector3f halfRes(res / 2, res / 2, res / 2);
    return gridIndex.cast<float>() * res + halfRes;
}
//##############################################################################
// Helper function to collect neighboring points
void lidarPreProcessing::collectNeighborPoints(const tsl::robin_map<Eigen::Vector3i, lidarPreProcessing::VoxelData,lidarPreProcessing::Vector3iHash, lidarPreProcessing::Vector3iEqual>& hashMap,
                                                const Eigen::Vector3i& gridIndex,
                                                const Eigen::Vector3f& queryPoint,
                                                std::vector<Eigen::Vector3f>& neighborPoints) 
{   
    // Iterate over the current voxel and its neighbors
    for (int x = -NUMNEIGHBORS; x <= NUMNEIGHBORS; ++x) {
        for (int y = -NUMNEIGHBORS; y <= NUMNEIGHBORS; ++y) {
            for (int z = -NUMNEIGHBORS; z <= NUMNEIGHBORS; ++z) {
                Eigen::Vector3i neighborIndex = gridIndex + Eigen::Vector3i(x, y, z);
                auto it = hashMap.find(neighborIndex);
                if (it == hashMap.end())
                    continue; // Skip if neighbor voxel doesn't exist

                const auto& neighborVoxel = it->second;
                for (const auto& neighborPoint : neighborVoxel.datas) {
                    // Add points to the neighbor list
                    neighborPoints.push_back(neighborPoint.position);
                }
            }
        }
    }
}
//##############################################################################
// Convert grid index back to world position (center of voxel)
void lidarPreProcessing::calculateSmoothness(
    tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual>& hashMap) 
{
    // Use TBB to parallelize processing of each voxel in the hash map
    tbb::parallel_for_each(hashMap.begin(), hashMap.end(), [&](auto& hashMapEntry) {
        auto& [gridIndex, voxel] = hashMapEntry;

        // Process each point in the voxel
        for (auto& PointData : voxel.datas) {
            const Eigen::Vector3f& queryPoint = PointData.position;
            float queryRange = queryPoint.norm();
            float queryRangeSquared = queryRange * queryRange;

            // Gather neighboring points
            std::vector<Eigen::Vector3f> neighborPoints;
            collectNeighborPoints(hashMap, gridIndex, queryPoint, neighborPoints, NUMNEIGHBORS);

            // Mark point as isolated if fewer than NUMNEIGHBORS neighbors
            if (neighborPoints.size() < NUMNEIGHBORS) {
                PointData.isIsolated = true;
                PointData.smoothness = std::numeric_limits<float>::max(); // Assign a default high value
                continue;
            }

            // Find the NUMNEIGHBORS nearest neighbors
            std::nth_element(neighborPoints.begin(), neighborPoints.begin() + NUMNEIGHBORS, neighborPoints.end(),
                [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
                    return (a - queryPoint).squaredNorm() < (b - queryPoint).squaredNorm();
                });

            // Compute smoothness using the NUMNEIGHBORS nearest neighbors
            float smoothness = 0.0f;
            for (int i = 0; i < NUMNEIGHBORS; ++i) {
                float neighborRange = neighborPoints[i].norm();
                smoothness += queryRangeSquared - 2 * queryRange * neighborRange + neighborRange * neighborRange;
            }

            PointData.smoothness = smoothness; // Update smoothness
        }
    });
}




