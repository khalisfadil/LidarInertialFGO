#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <atomic>

namespace slam {
    // Grid cell key for spatial partitioning
    struct CellKey {
        int x, y, z;

        CellKey() : x(0), y(0), z(0) {}  // Default constructor
        CellKey(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}  // Constructor for three ints

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
            // Use 64-bit integers for intermediate calculations
            size_t hash = 0xCBF29CE484222325ULL; // FNV-64 prime as initial seed
            constexpr size_t prime = 0x100000001B3ULL; // Another FNV-64 prime for mixing

            // Mix x
            hash ^= static_cast<size_t>(std::hash<int>{}(k.x));
            hash *= prime;

            // Mix y
            hash ^= static_cast<size_t>(std::hash<int>{}(k.y));
            hash *= prime;

            // Mix z
            hash ^= static_cast<size_t>(std::hash<int>{}(k.z));
            hash *= prime;

            // Finalize with additional mixing to improve avalanche effect
            hash ^= hash >> 33;
            hash *= 0xFF51AFD7ED558CCDULL; // Another prime constant
            hash ^= hash >> 33;
            hash *= 0xC4CEB9FE1A85EC53ULL; // Yet another prime constant
            hash ^= hash >> 33;

            return hash;
        }
    };

    struct PairHash {
        size_t operator()(const std::pair<CellKey, CellKey>& p) const {
            CellKeyHash hasher;
            size_t hash = 0xCBF29CE484222325ULL;
            constexpr size_t prime = 0x100000001B3ULL;

            hash ^= hasher(p.first);
            hash *= prime;
            hash ^= hasher(p.second);
            hash *= prime;

            hash ^= hash >> 33;
            hash *= 0xFF51AFD7ED558CCDULL;
            hash ^= hash >> 33;
            hash *= 0xC4CEB9FE1A85EC53ULL;
            hash ^= hash >> 33;

            return hash;
        }
    };

    struct PairEqual {
        bool operator()(const std::pair<CellKey, CellKey>& a, const std::pair<CellKey, CellKey>& b) const {
            return a.first == b.first && a.second == b.second;
        }
    };

}  // namespace slam