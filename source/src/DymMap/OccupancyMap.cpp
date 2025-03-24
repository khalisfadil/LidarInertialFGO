#include "DymMap/OccupancyMap.hpp"

namespace slam {
    namespace occmap {
        
        // -----------------------------------------------------------------------------
        // Section: OccupancyMap
        // -----------------------------------------------------------------------------

        OccupancyMap::OccupancyMap(double resolution, 
                                    double mapMaxDistance,
                                    Eigen::Vector3d mapOrigin, 
                                    unsigned int maxPointsPerVoxel, 
                                    ColorMode colorMode)
            : resolution_(resolution), 
            mapMaxDistance_(mapMaxDistance), 
            mapOrigin_(mapOrigin), 
            maxPointsPerVoxel_(maxPointsPerVoxel), 
            colorMode_(colorMode) {
        }

        // -----------------------------------------------------------------------------
        // Section: occupancyMap
        // -----------------------------------------------------------------------------

        void OccupancyMap::occupancyMap(const OccupancyMapDataFrame& frame) {
            if (frame.pointcloud.empty()) return;
            const auto& points = frame.pointcloud;
            occupancyMapBase(points, frame.frameID, frame.timestamp);
            clearUnwantedVoxel(frame.vehiclePosition);
        }

        // -----------------------------------------------------------------------------
        // Section: occupancyMapBase
        // -----------------------------------------------------------------------------

        void OccupancyMap::occupancyMapBase(const std::vector<Point3D>& points, unsigned int frame_id, double timestamp) {
            tracked_cell.clear();
            if (points.empty()) return;

            std::vector<CellKey> point_to_cell(points.size());

            tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()),
                [&](const tbb::blocked_range<size_t>& range) {
                    tbb::concurrent_unordered_set<CellKey, CellKeyHash> local_tracked;
                    local_tracked.reserve(range.size());

                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        CellKey key = CellKey::fromPoint(points[i].Pt, mapOrigin_, resolution_);
                        point_to_cell[i] = key;

                        auto result = occupancyMap_.insert({key, Voxel3D{}});
                        Voxel3D& voxel = result.first->second;

                        if (voxel.counter < maxPointsPerVoxel_) {
                            unsigned int newCount = voxel.counter + 1;
                            voxel.averagePointPose = (voxel.averagePointPose * voxel.counter + points[i].Pt) / newCount;
                            voxel.averagePointAtt = (voxel.averagePointAtt * voxel.counter + points[i].Att) / newCount;
                            voxel.counter = newCount;
                            voxel.frameID = frame_id;
                            voxel.timestamp = timestamp;
                            voxel.key = key;

                            // Switch based on color mode
                            switch (colorMode_) {
                                case ColorMode::Occupancy:
                                    voxel.color = computeOccupancyColor(newCount);
                                    break;
                                case ColorMode::Reflectivity:
                                    voxel.color = computeReflectivityColor(voxel.averagePointAtt.x());
                                    break;
                                case ColorMode::Intensity:
                                    voxel.color = computeIntensityColor(voxel.averagePointAtt.y());
                                    break;
                                case ColorMode::NIR:
                                    voxel.color = computeNIRColor(voxel.averagePointAtt.z());
                                    break;
                                default:  // Shouldn’t happen, but default to Occupancy
                                    voxel.color = computeOccupancyColor(newCount);
                                    break;
                            }

                            
                        } else {
                            voxel.frameID = frame_id;
                            voxel.timestamp = timestamp;
                        }
                        local_tracked.insert(key);
                    }
                    tracked_cell.insert(local_tracked.begin(), local_tracked.end());
                });
        }

        // -----------------------------------------------------------------------------
        // Section: clearUnwantedVoxel
        // -----------------------------------------------------------------------------

        void OccupancyMap::clearUnwantedVoxel(Eigen::Vector3d vehiclePosition) {
            using RaycastCache = tbb::concurrent_unordered_map<std::pair<CellKey, CellKey>, 
                                                              std::vector<CellKey>, PairHash, PairEqual>;
            RaycastCache raycastCache;
            raycastCache.reserve(tracked_cell.size());

            tbb::concurrent_unordered_set<CellKey, CellKeyHash> cellsToRemove;
            cellsToRemove.reserve(occupancyMap_.size() / 10);

            tbb::parallel_invoke(
                [&]() {
                    tbb::parallel_for_each(occupancyMap_.begin(), occupancyMap_.end(), 
                        [&](const auto& mapEntry) {
                            const Eigen::Vector3d voxelPos = mapEntry.second.computeGridToWorld(mapOrigin_, resolution_);
                            if ((voxelPos - vehiclePosition).squaredNorm() > mapMaxDistance_ * mapMaxDistance_) {
                                cellsToRemove.insert(mapEntry.first);
                            }
                        });
                },
                [&]() {
                    CellKey sourceCell = CellKey::fromPoint(vehiclePosition, mapOrigin_, resolution_);
                    tbb::parallel_for_each(tracked_cell.begin(), tracked_cell.end(),
                        [&](const CellKey& targetCell) {
                            auto result = raycastCache.insert(
                                {std::pair<CellKey, CellKey>{sourceCell, targetCell}, 
                                 performRaycast(sourceCell, targetCell)}
                            );
                            cellsToRemove.insert(result.first->second.begin(), result.first->second.end());
                        });
                }
            );

            tbb::parallel_for_each(cellsToRemove.begin(), cellsToRemove.end(),
                [&](const CellKey& key) {
                    occupancyMap_.unsafe_erase(key);
                });
        }

        // -----------------------------------------------------------------------------
        // Section: performRaycast
        // -----------------------------------------------------------------------------

        std::vector<CellKey> OccupancyMap::performRaycast(const CellKey& start, const CellKey& end) const {
            if (start == end) return {};
            Eigen::Vector3i startVec(start.x, start.y, start.z);
            Eigen::Vector3i endVec(end.x, end.y, end.z);
            Eigen::Vector3i diff = endVec - startVec;
            if (diff.squaredNorm() <= 1) return {};

            Eigen::Vector3i current = startVec;
            Eigen::Vector3i delta = diff.cwiseAbs();
            Eigen::Vector3i step = diff.cwiseSign();

            std::vector<CellKey> voxelIndices;
            voxelIndices.reserve(delta.maxCoeff());

            voxelIndices.emplace_back(current.x(), current.y(), current.z());

            int primaryAxis = (delta.x() >= delta.y() && delta.x() >= delta.z()) ? 0 :
                             (delta.y() >= delta.z()) ? 1 : 2;
            int secondaryAxis = (primaryAxis + 1) % 3;
            int tertiaryAxis = (primaryAxis + 2) % 3;

            int error1 = 2 * delta[secondaryAxis] - delta[primaryAxis];
            int error2 = 2 * delta[tertiaryAxis] - delta[primaryAxis];

            Eigen::Vector3i next = current;
            while (true) {
                if (delta[primaryAxis] > 0) next[primaryAxis] = current[primaryAxis] + step[primaryAxis];
                if (error1 > 0) {
                    next[secondaryAxis] = current[secondaryAxis] + step[secondaryAxis];
                    error1 -= 2 * delta[primaryAxis];
                }
                if (error2 > 0) {
                    next[tertiaryAxis] = current[tertiaryAxis] + step[tertiaryAxis];
                    error2 -= 2 * delta[primaryAxis];
                }
                error1 += 2 * delta[secondaryAxis];
                error2 += 2 * delta[tertiaryAxis];

                if (next == endVec) break;

                current = next;
                voxelIndices.emplace_back(current.x(), current.y(), current.z());
            }

            return voxelIndices;
        }

        // -----------------------------------------------------------------------------
        // Section: assignVoxelColorsRed
        // -----------------------------------------------------------------------------

        Eigen::Vector3i OccupancyMap::computeOccupancyColor(unsigned int counter) const {
            int value = (255 * std::min(counter, maxPointsPerVoxel_)) / maxPointsPerVoxel_;
            return Eigen::Vector3i(value, value, value);
        }

        // -----------------------------------------------------------------------------
        // Section: assignVoxelColorsRed
        // -----------------------------------------------------------------------------

        Eigen::Vector3i OccupancyMap::computeReflectivityColor(double avgReflectivity) const{
            int reflectivityColorValue;
            if (avgReflectivity <= 100.0) {
                reflectivityColorValue = static_cast<int>(avgReflectivity * 2.55);
            } else {
                float transitionFactor = 0.2;
                if (avgReflectivity <= 110.0) {
                    float linearComponent = 2.55 * avgReflectivity;
                    float logComponent = 155.0 + (100.0 * (std::log2(avgReflectivity - 100.0 + 1.0) / std::log2(156.0)));
                    reflectivityColorValue = static_cast<int>((1.0 - transitionFactor) * linearComponent + transitionFactor * logComponent);
                } else {
                    float logReflectivity = std::log2(avgReflectivity - 100.0 + 1.0) / std::log2(156.0);
                    reflectivityColorValue = static_cast<int>(155.0 + logReflectivity * 100.0);
                }
            }
            return Eigen::Vector3i(std::clamp(reflectivityColorValue, 0, 255), 
                                std::clamp(reflectivityColorValue, 0, 255), 
                                std::clamp(reflectivityColorValue, 0, 255));
        }

        // -----------------------------------------------------------------------------
        // Section: calculateIntensityColor
        // -----------------------------------------------------------------------------

        Eigen::Vector3i OccupancyMap::computeIntensityColor(double avgIntensity) const{
            int intensityColorValue = static_cast<int>(std::clamp(avgIntensity, 0.0, 255.0));
            return Eigen::Vector3i(intensityColorValue, intensityColorValue, intensityColorValue);
        }

        // -----------------------------------------------------------------------------
        // Section: calculateNIRColor
        // -----------------------------------------------------------------------------

        Eigen::Vector3i OccupancyMap::computeNIRColor(double avgNIR) const{
            int NIRColorValue = static_cast<int>(std::clamp(avgNIR, 0.0, 255.0));
            return Eigen::Vector3i(NIRColorValue, NIRColorValue, NIRColorValue);
        }

        // -----------------------------------------------------------------------------
        // Section: calculateNIRColor
        // -----------------------------------------------------------------------------

        std::vector<Voxel3D> OccupancyMap::getOccupiedVoxel() const {
            std::vector<Voxel3D> occupiedVoxels;
            // Reserve space based on current map size to minimize reallocations
            occupiedVoxels.reserve(occupancyMap_.size());

            // Parallel iteration over the concurrent map
            tbb::parallel_for_each(occupancyMap_.begin(), occupancyMap_.end(),
                [&](const auto& mapEntry) {
                    const Voxel3D& voxel = mapEntry.second;
                    // Only include voxels that have points (counter > 0)
                    if (voxel.counter > 0) {
                        // Thread-safe push_back since we're using parallel_for_each
                        occupiedVoxels.push_back(voxel);
                    }
                });

            // Shrink to fit the actual size (optional, for memory efficiency)
            occupiedVoxels.shrink_to_fit();
            return occupiedVoxels;
        }



    }  // namespace occmap
}  // namespace slam