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
            cellsToRemove.reserve(occupancyMap_.size() / 2);

            auto distanceCheck = [&]() {
                tbb::parallel_for_each(occupancyMap_.begin(), occupancyMap_.end(), 
                    [&](const auto& mapEntry) {
                        const Eigen::Vector3d voxelPos = mapEntry.second.computeGridToWorld(mapOrigin_, resolution_);
                        if ((voxelPos - vehiclePosition).squaredNorm() > mapMaxDistance_ * mapMaxDistance_) {
                            cellsToRemove.insert(mapEntry.first);
                        }
                    });
            };

            CellKey sourceCell = CellKey::fromPoint(vehiclePosition, mapOrigin_, resolution_);
            auto raycasting = [&]() {
                tbb::parallel_for_each(tracked_cell.begin(), tracked_cell.end(),
                    [&](const CellKey& targetCell) {
                        auto result = raycastCache.insert(
                            {std::pair<CellKey, CellKey>{sourceCell, targetCell}, 
                            performRaycast(sourceCell, targetCell)}
                        );
                        cellsToRemove.insert(result.first->second.begin(), result.first->second.end());
                    });
            };

            if (occupancyMap_.size() > PARALLEL_THRESHOLD) {
                tbb::parallel_invoke(distanceCheck, raycasting);
            } else {
                distanceCheck();
                raycasting();
            }

            // Rebuild new map
            tbb::concurrent_unordered_map<CellKey, Voxel3D, CellKeyHash> newOccupancyMap;
            newOccupancyMap.reserve(occupancyMap_.size() - cellsToRemove.size());
            tbb::parallel_for_each(occupancyMap_.begin(), occupancyMap_.end(),
                [&](const auto& mapEntry) {
                    if (cellsToRemove.find(mapEntry.first) == cellsToRemove.end()) {
                        newOccupancyMap.insert(mapEntry);
                    }
                });

            // Swap safely
            occupancyMap_.swap(newOccupancyMap);
        }

        // -----------------------------------------------------------------------------
        // Section: performRaycast
        // -----------------------------------------------------------------------------

        std::vector<CellKey> OccupancyMap::performRaycast(const CellKey& start, const CellKey& end) const {
            // If start and end points are the same, return the starting voxel only
            if (start == end) {
                return {start};
            }

            // Convert CellKey to Eigen::Vector3i for computation
            Eigen::Vector3i startVoxel(start.x, start.y, start.z);
            Eigen::Vector3i endVoxel(end.x, end.y, end.z);

            // Initialize voxel collection using a concurrent container
            tbb::concurrent_vector<CellKey> voxelIndices;

            // Initialize Bresenham's algorithm parameters
            Eigen::Vector3i delta = (endVoxel - startVoxel).cwiseAbs();
            Eigen::Vector3i step = (endVoxel - startVoxel).cwiseSign();
            Eigen::Vector3i currentVoxel = startVoxel;

            voxelIndices.push_back(start); // Include the starting voxel as CellKey

            // Determine the dominant axis
            // int maxAxis = delta.maxCoeff();
            int primaryAxis = (delta.x() >= delta.y() && delta.x() >= delta.z()) ? 0 : (delta.y() >= delta.z() ? 1 : 2);

            // Initialize error terms for Bresenham's algorithm
            int error1 = 2 * delta[(primaryAxis + 1) % 3] - delta[primaryAxis];
            int error2 = 2 * delta[(primaryAxis + 2) % 3] - delta[primaryAxis];

            // Traverse the line using Bresenham's algorithm, stopping before reaching endVoxel
            for (int i = 0; i < delta[primaryAxis]; ++i) {
                currentVoxel[primaryAxis] += step[primaryAxis];

                if (error1 > 0) {
                    currentVoxel[(primaryAxis + 1) % 3] += step[(primaryAxis + 1) % 3];
                    error1 -= 2 * delta[primaryAxis];
                }

                if (error2 > 0) {
                    currentVoxel[(primaryAxis + 2) % 3] += step[(primaryAxis + 2) % 3];
                    error2 -= 2 * delta[primaryAxis];
                }

                error1 += 2 * delta[(primaryAxis + 1) % 3];
                error2 += 2 * delta[(primaryAxis + 2) % 3];

                // Stop if the next voxel would be the endVoxel
                if (currentVoxel == endVoxel) {
                    break;
                }

                // Add the current voxel to the result as CellKey
                voxelIndices.push_back(CellKey(currentVoxel.x(), currentVoxel.y(), currentVoxel.z()));
            }

            // Return results as a standard vector of CellKey
            return std::vector<CellKey>(voxelIndices.begin(), voxelIndices.end());
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
            // Use a tbb::concurrent_vector for thread-safe parallel collection
            tbb::concurrent_vector<Voxel3D> occupiedVoxels;
            occupiedVoxels.reserve(occupancyMap_.size()); // Reserve space to reduce reallocations

            // Parallel iteration over the concurrent map
            tbb::parallel_for_each(occupancyMap_.begin(), occupancyMap_.end(),
                [&](const auto& mapEntry) {
                    const Voxel3D& voxel = mapEntry.second;
                    if (voxel.counter > 0) {
                        occupiedVoxels.push_back(voxel); // Thread-safe push_back
                    }
                });

            // Convert to std::vector and return
            return std::vector<Voxel3D>(occupiedVoxels.begin(), occupiedVoxels.end());
        }
    }  // namespace occmap
}  // namespace slam