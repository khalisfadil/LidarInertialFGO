#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <deque>
#include <map>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_for_each.h>

#include "Utils/cluster.hpp"
#include "Utils/dataframe.hpp"
#include "Utils/cellKey.hpp"
#include "Utils/voxel.hpp"
#include "Utils/colorMode.hpp"
#include "DymCluster/Hungarian.hpp"

namespace slam {
    namespace cluster {

        class UnionFind {
        public:
            UnionFind(size_t n) : parent_(n), rank_(n, 0) {
                std::iota(parent_.begin(), parent_.end(), 0);
            }

            size_t find(size_t x) {
                if (parent_[x] != x) {
                    parent_[x] = find(parent_[x]);
                }
                return parent_[x];
            }

            void unite(size_t x, size_t y) {
                size_t rx = find(x);
                size_t ry = find(y);
                if (rx == ry) return;
                if (rank_[rx] < rank_[ry]) std::swap(rx, ry);
                parent_[ry] = rx;
                if (rank_[rx] == rank_[ry]) ++rank_[rx];
            }

        private:
            std::vector<size_t> parent_;
            std::vector<size_t> rank_;
        };

        using stateID = unsigned int;

        inline stateID NewStateID() {
            static std::atomic<unsigned int> id{0};
            return id.fetch_add(1, std::memory_order_relaxed);
        }

        class ClusterExtraction {
        public:
            ClusterExtraction(double resolution = 0.1,
                              double tolerance = 0.1,
                              size_t min_size = 3,
                              size_t max_size = 1000,
                              size_t max_frames = 5,
                              unsigned int maxPointsPerVoxel = 20,
                              ColorMode colorMode = ColorMode::Occupancy)
                : resolution_(resolution), cluster_tolerance_(tolerance), min_cluster_size_(min_size),
                  max_cluster_size_(max_size), max_frames_(max_frames), mapOrigin_(Eigen::Vector3d::Zero()),
                  maxPointsPerVoxel_(maxPointsPerVoxel), colorMode_(colorMode) {}

            void extractClusters(const ClusterExtractorDataFrame& frame);

            const std::vector<slam::Cluster3D>& getClusters() const { return clusters_; }

            std::vector<Voxel3D> getOccupiedVoxel() const;

        private:
            void extractBaseClusters(const std::vector<Point3D>& points, unsigned int frame_id, double timestamp);
            void clusterOccupancyMapBase(const std::vector<Point3D>& points, unsigned int frame_id, double timestamp);
            double calculateCost(const slam::Cluster3D& clusterA, const slam::Cluster3D& clusterB) const;
            double cauchyCost(double error_norm, double k = 2.0);
            void optimizeCentroid(const std::map<double, Eigen::Vector3d>& prev_states,
                                  Eigen::Vector3d& centroid,
                                  const Eigen::Matrix3d& L,
                                  double timestamp,
                                  double& final_cost,
                                  int max_iterations = 10);
            Eigen::Vector3i computeOccupancyColor(unsigned int counter) const;
            Eigen::Vector3i computeReflectivityColor(double avgReflectivity) const;
            Eigen::Vector3i computeIntensityColor(double avgIntensity) const;
            Eigen::Vector3i computeNIRColor(double avgNIR) const;

            using GridType = tbb::concurrent_unordered_map<CellKey, Voxel3D, CellKeyHash>;
            GridType occupancyMap_;

            double resolution_;
            double cluster_tolerance_;
            size_t min_cluster_size_;
            size_t max_cluster_size_;
            size_t max_frames_;
            std::vector<slam::Cluster3D> clusters_;
            std::deque<std::vector<slam::Cluster3D>> prevClusters_;
            double max_distance_threshold_ = 2.0;
            std::vector<slam::Point3D> dynamic_points_;
            const Eigen::Vector3d mapOrigin_;
            const unsigned int maxPointsPerVoxel_;
            const ColorMode colorMode_;
        };
    }  // namespace cluster
}  // namespace slam