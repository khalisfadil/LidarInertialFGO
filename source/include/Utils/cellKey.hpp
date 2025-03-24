#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <atomic>

namespace slam {
    // Grid cell key for spatial partitioning
    struct CellKey {
        int x, y, z;

        bool operator==(const CellKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }

        static CellKey fromPoint(const Eigen::Vector3d& point, 
                                const Eigen::Vector3d& origin, 
                                double resolution) {
            return {
                static_cast<int>(std::floor((point.x() - origin.x()) / resolution)),
                static_cast<int>(std::floor((point.y() - origin.y()) / resolution)),
                static_cast<int>(std::floor((point.z() - origin.z()) / resolution))
            };
        }
    };

    // Hash function for CellKey
    struct CellKeyHash {
        size_t operator()(const CellKey& k) const {
            return std::hash<int>{}(k.x) ^ 
                (std::hash<int>{}(k.y) << 1) ^ 
                (std::hash<int>{}(k.z) << 2);
        }
    };

}  // namespace slam

