#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <numeric>
#include <limits>
#include <algorithm>
#include <cmath>
#include <mutex>  // Include for std::mutex
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_for_each.h>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_set.h>

#include "Utils/dataframe.hpp"
#include "Utils/cellKey.hpp"
#include "Utils/voxel.hpp"
#include "Utils/colorMode.hpp"
#include "Utils/constants.hpp"
#include "Utils/mapconfig.hpp"

namespace slam {
    namespace occmap {

        class OccupancyMap {
        public:
            OccupancyMap(double resolution = 0.1, double mapMaxDistance = 500.0,
                         Eigen::Vector3d mapOrigin = Eigen::Vector3d::Zero(),
                         unsigned int maxPointsPerVoxel = 20,
                         ColorMode colorMode = ColorMode::Occupancy);

            void occupancyMap(const OccupancyMapDataFrame& frame);
            std::vector<Voxel3D> getOccupiedVoxel() const;  

        private:
            std::mutex mapMutex_;  // No need for mutable

            alignas(64) MapConfig mapConfig_;

            void occupancyMapBase(const std::vector<Point3D>& points, unsigned int frame_id, double timestamp);
            void clearUnwantedVoxel(const Eigen::Vector3d& vehiclePosition);
            std::vector<CellKey> performRaycast(const CellKey& start, const CellKey& end) const;
            Eigen::Vector3i computeOccupancyColor(unsigned int counter) const;
            Eigen::Vector3i computeReflectivityColor(double avgReflectivity) const;
            Eigen::Vector3i computeIntensityColor(double avgIntensity) const;
            Eigen::Vector3i computeNIRColor(double avgNIR) const;

            const double resolution_;
            const double mapMaxDistance_;
            const Eigen::Vector3d mapOrigin_;
            const unsigned int maxPointsPerVoxel_;
            const ColorMode colorMode_;

            using GridType = tbb::concurrent_unordered_map<CellKey, Voxel3D, CellKeyHash>;
            GridType occupancyMap_;
            tbb::concurrent_unordered_set<CellKey, CellKeyHash> tracked_cell;
        };

    }  // namespace occmap
}  // namespace slam