#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <boost/asio.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <tbb/parallel_invoke.h>
#include "DymMap/OccupancyMap.hpp"
#include "DymCluster/ClusterExtraction.hpp"
#include "Utils/Callback/callbackPoints.hpp"
#include "Utils/mapconfig.hpp"
#include "Utils/UDPpacket/udpSocket.hpp"
#include "Utils/dataframe.hpp"
#include "Utils/voxel.hpp"
#include "Utils/Open3d/voxelGrid.hpp"

namespace slam { // Opening namespace brace

    class Pipeline { // Opening class brace
    public:
        static Pipeline& getInstance() noexcept;
        Pipeline(const Pipeline&) = delete;
        Pipeline& operator=(const Pipeline&) = delete;

        inline static std::unique_ptr<occmap::OccupancyMap> occupancyMapInstance = nullptr;
        inline static std::unique_ptr<cluster::ClusterExtraction> clusterExtractionInstance = nullptr;
        static boost::lockfree::spsc_queue<VehiclePoseDataFrame, boost::lockfree::capacity<128>> ringBufferPose;
        static boost::lockfree::spsc_queue<OccupancyMapDataFrame, boost::lockfree::capacity<128>> pointsRingBufferOccMap;
        static boost::lockfree::spsc_queue<ClusterExtractorDataFrame, boost::lockfree::capacity<128>> pointsRingBufferExtCls;
        static boost::lockfree::spsc_queue<std::vector<Voxel3D>, boost::lockfree::capacity<128>> voxelsRingBufferOccMap;
        static boost::lockfree::spsc_queue<std::vector<Voxel3D>, boost::lockfree::capacity<128>> voxelsRingBufferExtCls;
        static boost::lockfree::spsc_queue<std::string, boost::lockfree::capacity<1024>> logQueue;
        static std::atomic<bool> running;
        static std::atomic<int> droppedLogs;
        static std::condition_variable globalCV;

        static void signalHandler(int signal) noexcept;
        void startPointsListener(boost::asio::io_context& ioContext,
                                 std::string_view host,
                                 uint16_t port,
                                 uint32_t bufferSize,
                                 const std::vector<int>& allowedCores) noexcept;
        void setThreadAffinity(const std::vector<int>& coreIDs) noexcept;
        void runOccupancyMapPipeline(const std::vector<int>& allowedCores) noexcept;
        void runClusterExtractionPipeline(const std::vector<int>& allowedCores) noexcept;
        void runVizualizationPipeline(const std::vector<int>& allowedCores) noexcept;
        bool updateVisualization(open3d::visualization::Visualizer* vis) noexcept;
        void processLogQueue(const std::vector<int>& allowedCores) noexcept;

    private:
        Pipeline();
        alignas(64) MapConfig mapConfig_;
        alignas(64) ProcessConfig processConfig_;
        static std::thread logThread_;
        
        static std::shared_ptr<open3d::geometry::VoxelGrid> voxel_grid_occMap_ptr;
        static std::shared_ptr<open3d::geometry::VoxelGrid> voxel_grid_extCls_ptr;
        static std::shared_ptr<open3d::geometry::TriangleMesh> vehicle_mesh_ptr;
    }; // Closing class brace

} // Closing namespace brace