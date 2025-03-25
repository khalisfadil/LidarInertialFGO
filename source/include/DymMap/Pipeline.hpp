#pragma once

#include <memory>
#include <vector>
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

namespace slam {
    class Pipeline {
    public:
        Pipeline();
        ~Pipeline();

        Pipeline(const Pipeline&) = delete;
        Pipeline& operator=(const Pipeline&) = delete;

        static void signalHandler(int signal);  // Keep static for signal handling
        void startPointsListener(boost::asio::io_context& ioContext, 
                               const std::string& host, 
                               uint16_t port,
                               uint32_t bufferSize, 
                               const std::vector<int>& allowedCores);
        void runOccupancyMapPipeline(const std::vector<int>& allowedCores);
        void runClusterExtractionPipeline(const std::vector<int>& allowedCores);
        void runVizualizationPipeline(const std::vector<int>& allowedCores);
        bool updateVisualization(open3d::visualization::Visualizer* vis);
        void processLogQueue(const std::vector<int>& allowedCores);
        void processReportQueueOccMap(const std::string& filename, const std::vector<int>& allowedCores);
        void processReportQueueExtCls(const std::string& filename, const std::vector<int>& allowedCores);

        // Public access to running_ for main loop
        std::atomic<bool> running_{true};

    private:
        void setThreadAffinity(const std::vector<int>& coreIDs);

        std::unique_ptr<occmap::OccupancyMap> occupancyMapInstance_;
        std::unique_ptr<cluster::ClusterExtraction> clusterExtractionInstance_;
        boost::lockfree::spsc_queue<VehiclePoseDataFrame, boost::lockfree::capacity<128>> ringBufferPose_;
        boost::lockfree::spsc_queue<OccupancyMapDataFrame, boost::lockfree::capacity<128>> pointsRingBufferOccMap_;
        boost::lockfree::spsc_queue<ClusterExtractorDataFrame, boost::lockfree::capacity<128>> pointsRingBufferExtCls_;
        boost::lockfree::spsc_queue<std::vector<Voxel3D>, boost::lockfree::capacity<128>> voxelsRingBufferOccMap_;
        boost::lockfree::spsc_queue<std::vector<Voxel3D>, boost::lockfree::capacity<128>> voxelsRingBufferExtCls_;
        boost::lockfree::spsc_queue<std::string, boost::lockfree::capacity<1024>> logQueue_;
        boost::lockfree::spsc_queue<ReportDataFrame, boost::lockfree::capacity<1024>> reportOccupancyMapQueue_;
        boost::lockfree::spsc_queue<ReportDataFrame, boost::lockfree::capacity<1024>> reportExtractClusterQueue_;

        std::atomic<int> droppedLogs_{0};
        std::atomic<int> droppedOccupancyMapReports_{0};
        std::atomic<int> droppedExtractClusterReports_{0};
        std::condition_variable globalCV_;

        alignas(64) MapConfig mapConfig_;
        alignas(64) ProcessConfig processConfig_;
        CallbackPoints callbackPointsProcessor_;

        std::shared_ptr<open3d::geometry::VoxelGrid> voxel_grid_occMap_ptr_;
        std::shared_ptr<open3d::geometry::VoxelGrid> voxel_grid_extCls_ptr_;
        std::shared_ptr<open3d::geometry::TriangleMesh> vehicle_mesh_ptr_;
    };
} // namespace slam