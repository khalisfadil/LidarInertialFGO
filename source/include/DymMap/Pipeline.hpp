#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <iostream>
#include <fstream>

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

        static std::unique_ptr<occmap::OccupancyMap> occupancyMapInstance;
        static std::unique_ptr<cluster::ClusterExtraction> clusterExtractionInstance;
        static boost::lockfree::spsc_queue<VehiclePoseDataFrame, boost::lockfree::capacity<128>> ringBufferPose;
        static boost::lockfree::spsc_queue<OccupancyMapDataFrame, boost::lockfree::capacity<128>> pointsRingBufferOccMap;
        static boost::lockfree::spsc_queue<ClusterExtractorDataFrame, boost::lockfree::capacity<128>> pointsRingBufferExtCls;
        static boost::lockfree::spsc_queue<std::vector<Voxel3D>, boost::lockfree::capacity<128>> voxelsRingBufferOccMap;
        static boost::lockfree::spsc_queue<std::vector<Voxel3D>, boost::lockfree::capacity<128>> voxelsRingBufferExtCls;
        static boost::lockfree::spsc_queue<std::string, boost::lockfree::capacity<1024>> logQueue;
        static boost::lockfree::spsc_queue<ReportDataFrame, boost::lockfree::capacity<1024>> reportOccupancyMapQueue;
        static boost::lockfree::spsc_queue<ReportDataFrame, boost::lockfree::capacity<1024>> reportExtractClusterQueue;

        static std::atomic<bool> running;
        static std::atomic<int> droppedLogs;
        static std::atomic<int> droppedOccupancyMapReports;
        static std::atomic<int> droppedExtractClusterReports;
        static std::condition_variable globalCV;

        Pipeline();
        static void signalHandler(int signal) ;
        void startPointsListener(boost::asio::io_context& ioContext, 
                                    const std::string& host, 
                                    uint16_t port,
                                    uint32_t bufferSize, 
                                    const std::vector<int>& allowedCores);

        void setThreadAffinity(const std::vector<int>& coreIDs) ;
        void runOccupancyMapPipeline(const std::vector<int>& allowedCores) ;
        void runClusterExtractionPipeline(const std::vector<int>& allowedCores) ;
        void runVizualizationPipeline(const std::vector<int>& allowedCores) ;
        bool updateVisualization(open3d::visualization::Visualizer* vis) ;
        void processLogQueue(const std::vector<int>& allowedCores) ;
        void processReportQueueOccMap(const std::string& filename, const std::vector<int>& allowedCores) ;
        void processReportQueueExtCls(const std::string& filename, const std::vector<int>& allowedCores) ;

    private:

        alignas(64) MapConfig mapConfig_;
        alignas(64) ProcessConfig processConfig_;
        static std::thread logThread_;

        CallbackPoints callbackPointsProcessor;
        
        static std::shared_ptr<open3d::geometry::VoxelGrid> voxel_grid_occMap_ptr;
        static std::shared_ptr<open3d::geometry::VoxelGrid> voxel_grid_extCls_ptr;
        static std::shared_ptr<open3d::geometry::TriangleMesh> vehicle_mesh_ptr;
    }; // Closing class brace

} // Closing namespace brace