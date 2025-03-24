#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <numeric>
#include <limits>
#include <algorithm>
#include <cmath>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_for_each.h>
#include <tbb/concurrent_unordered_set.h>

#include "Utils/dataframe.hpp"
#include "Utils/cellKey.hpp"
#include "Utils/voxel.hpp"
#include "Utils/colorMode.hpp"

namespace slam {
    namespace occmap {
        class OccupancyMap {
        public:
            OccupancyMap(double resolution = 0.1, double mapMaxDistance = 500.0,
                         Eigen::Vector3d mapOrigin = Eigen::Vector3d::Zero(),
                         unsigned int maxPointsPerVoxel = 20,
                         ColorMode colorMode = ColorMode::Occupancy)
                : resolution_(resolution), mapMaxDistance_(mapMaxDistance), mapOrigin_(mapOrigin),
                  maxPointsPerVoxel_(maxPointsPerVoxel), colorMode_(colorMode),
                  vehiclePosition_(Eigen::Vector3d::Zero()) {}

            void occupancyMap(const OccupancyMapDataFrame& frame);

            std::vector<Voxel3D> getOccupiedVoxel() const;

        private:
            struct PairHash {
                size_t operator()(const std::pair<CellKey, CellKey>& p) const {
                    CellKeyHash hasher;
                    size_t h1 = hasher(p.first);
                    size_t h2 = hasher(p.second);
                    return h1 ^ (h2 << 1);
                }
            };

            struct PairEqual {
                bool operator()(const std::pair<CellKey, CellKey>& a, const std::pair<CellKey, CellKey>& b) const {
                    return a.first == b.first && a.second == b.second;
                }
            };

            void occupancyMapBase(const std::vector<Point3D>& points, unsigned int frame_id, double timestamp);
            void clearUnwantedVoxel(Eigen::Vector3d vehiclePosition);
            std::vector<CellKey> performRaycast(const CellKey& start, const CellKey& end) const;
            Eigen::Vector3i computeOccupancyColor(unsigned int counter) const;
            Eigen::Vector3i computeReflectivityColor(double avgReflectivity) const;
            Eigen::Vector3i computeIntensityColor(double avgIntensity) const;
            Eigen::Vector3i computeNIRColor(double avgNIR) const;

            using GridType = tbb::concurrent_unordered_map<CellKey, Voxel3D, CellKeyHash>;
            GridType occupancyMap_;
            tbb::concurrent_unordered_set<CellKey, CellKeyHash> tracked_cell;

            const double resolution_;
            const double mapMaxDistance_;
            const Eigen::Vector3d mapOrigin_;
            Eigen::Vector3d vehiclePosition_;
            const unsigned int maxPointsPerVoxel_;
            const ColorMode colorMode_;
        };
    }  // namespace occmap
}  // namespace slam