#pragma once

#include <cstdint>
#include <deque>
#include <algorithm>
#include <cstddef>

#include <Eigen/Dense>

#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>

constexpr int NUMNEIGHBORS = 5;
constexpr float MAPRES = 0.2;

class lidarPreProcessing{
    public:

    struct PointData {
            Eigen::Vector3f position;  // Position of the point
            Eigen::Vector3f attributes;
            mutable float smoothness;
            mutable bool isIsolated = false;
            PointData()
                : position(Eigen::Vector3f::Zero()),
                  attributes(Eigen::Vector3f::Zero()),
                  smoothness(std::numeric_limits<float>::max()),
                  isIsolated(false){}
    };

    struct VoxelData{
        std::vector<PointData> datas;
        Eigen::Vector3f centerPosition;
    };

    //##############################################################################
    // Custom hash function for Eigen::Vector3i
    struct Vector3iHash {
        std::size_t operator()(const Eigen::Vector3i& vec) const {
            return std::hash<int>()(vec.x()) ^ (std::hash<int>()(vec.y()) << 1) ^ (std::hash<int>()(vec.z()) << 2);}};
    //##############################################################################
    // Custom equality function for Eigen::Vector3i
    struct Vector3iEqual {
        bool operator()(const Eigen::Vector3i& lhs, const Eigen::Vector3i& rhs) const {
            return lhs == rhs;}};

    private:

    

    

    Eigen::Vector3i posToGridIndex(const Eigen::Vector3f& pos, const float& res) const;
    Eigen::Vector3f gridToWorld(const Eigen::Vector3i& gridIndex,const float& res) const;
    
    
    tsl::robin_map<Eigen::Vector3i, VoxelData, 
                    Vector3iHash, 
                    Vector3iEqual> constructHashMap(std::vector<Eigen::Vector3f> points_,
                                                                std::vector<Eigen::Vector3f> attributes_,
                                                                float res);

    void calculateSmoothness(tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual>& hashMap);

    void collectNeighborPoints(const tsl::robin_map<Eigen::Vector3i, lidarPreProcessing::VoxelData,lidarPreProcessing::Vector3iHash, lidarPreProcessing::Vector3iEqual>& hashMap,
                                                const Eigen::Vector3i& gridIndex,
                                                const Eigen::Vector3f& queryPoint,
                                                std::vector<Eigen::Vector3f>& neighborPoints);  


    void calculateSmoothness();
    void featureExtraction();
    
};