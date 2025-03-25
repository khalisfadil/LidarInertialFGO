#include "DymMap/Pipeline.hpp"

#include <iostream>
#include <csignal>
#include <cstring>
#include <unistd.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <sched.h>
#include <pthread.h>
#include <fstream>  // Added for std::ofstream

namespace slam {
    Pipeline::Pipeline() {
        occupancyMapInstance_ = std::make_unique<occmap::OccupancyMap>(
            mapConfig_.resolution, mapConfig_.mapMaxDistance, mapConfig_.mapOrigin,
            mapConfig_.maxPointsPerVoxel, mapConfig_.colorMode);
        clusterExtractionInstance_ = std::make_unique<cluster::ClusterExtraction>(
            mapConfig_.resolution, mapConfig_.mapOrigin, mapConfig_.tolerance, mapConfig_.min_size, mapConfig_.max_size,
            mapConfig_.max_frames, mapConfig_.maxPointsPerVoxel, mapConfig_.colorMode);
        voxel_grid_occMap_ptr_ = std::make_shared<open3d::geometry::VoxelGrid>();
        voxel_grid_extCls_ptr_ = std::make_shared<open3d::geometry::VoxelGrid>();
        vehicle_mesh_ptr_ = std::make_shared<open3d::geometry::TriangleMesh>();
    }

    Pipeline::~Pipeline() {
        running_.store(false, std::memory_order_release);
        globalCV_.notify_all();
    }

    void Pipeline::signalHandler(int signal) {
        if (signal == SIGINT || signal == SIGTERM) {
            // Note: Can't access instance members here since this is static
            // For simplicity, we'll assume a single instance and handle shutdown externally
            write(STDOUT_FILENO, "[signalHandler] Shutting down...\n", 33);
        }
    }

    void Pipeline::processLogQueue(const std::vector<int>& allowedCores) {
        setThreadAffinity(allowedCores);

        std::string message;
        int lastReportedDrops = 0;
        while (running_.load(std::memory_order_acquire)) {
            if (logQueue_.pop(message)) {
                std::cerr << message;
                int currentDrops = droppedLogs_.load(std::memory_order_relaxed);
                if (currentDrops > lastReportedDrops && (currentDrops - lastReportedDrops) >= 100) {
                    std::ostringstream oss;
                    oss << "[Logging] Warning: " << (currentDrops - lastReportedDrops) << " log messages dropped due to queue overflow.\n";
                    std::cerr << oss.str();
                    lastReportedDrops = currentDrops;
                }
            } else {
                std::this_thread::yield();
            }
        }
        while (logQueue_.pop(message)) {
            std::cerr << message;
        }
        int finalDrops = droppedLogs_.load(std::memory_order_relaxed);
        if (finalDrops > lastReportedDrops) {
            std::cerr << "[Logging] Final report: " << (finalDrops - lastReportedDrops) << " log messages dropped.\n";
        }
    }

    void Pipeline::processReportQueueOccMap(const std::string& filename, const std::vector<int>& allowedCores) {
        setThreadAffinity(allowedCores);

        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::ostringstream oss;
            oss << "[ReportWriter] Error: Failed to open file " << filename << " for writing.\n";
            if (!logQueue_.push(oss.str())) {
                droppedLogs_.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }

        outfile << "# Report Data\n";
        outfile << "# " << std::left << std::setw(10) << "FrameID" 
                << std::setw(20) << "Timestamp" 
                << std::setw(20) << "ElapsedTime" 
                << std::setw(15) << "NumPoints" 
                << std::setw(15) << "OccMapSize" << "\n";

        ReportDataFrame data;
        int lastReportedDrops = 0;

        while (running_.load(std::memory_order_acquire)) {
            if (reportOccupancyMapQueue_.pop(data)) {
                outfile << std::left << std::setw(10) << data.frameID
                        << std::fixed << std::setprecision(6) 
                        << std::setw(20) << data.timestamp
                        << std::setw(20) << data.elapsedTime
                        << std::setw(15) << data.numpoint
                        << std::setw(15) << data.occmapsize << "\n";

                int currentDrops = droppedOccupancyMapReports_.load(std::memory_order_relaxed);
                if (currentDrops > lastReportedDrops && (currentDrops - lastReportedDrops) >= 100) {
                    std::ostringstream oss;
                    oss << "[ReportWriter] Warning: " << (currentDrops - lastReportedDrops) 
                        << " occupancy map report entries dropped due to queue overflow.\n";
                    if (!logQueue_.push(oss.str())) {
                        droppedLogs_.fetch_add(1, std::memory_order_relaxed);
                    }
                    lastReportedDrops = currentDrops;
                }
            } else {
                std::this_thread::yield();
            }
        }

        while (reportOccupancyMapQueue_.pop(data)) {
            outfile << std::left << std::setw(10) << data.frameID
                    << std::fixed << std::setprecision(6) 
                    << std::setw(20) << data.timestamp
                    << std::setw(20) << data.elapsedTime
                    << std::setw(15) << data.numpoint
                    << std::setw(15) << data.occmapsize << "\n";
        }

        int finalDrops = droppedOccupancyMapReports_.load(std::memory_order_relaxed);
        if (finalDrops > lastReportedDrops) {
            std::ostringstream oss;
            oss << "[ReportWriter] Final report: " << (finalDrops - lastReportedDrops) 
                << " occupancy map report entries dropped.\n";
            if (!logQueue_.push(oss.str())) {
                droppedLogs_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        outfile.flush();
        outfile.close();
    }

    void Pipeline::processReportQueueExtCls(const std::string& filename, const std::vector<int>& allowedCores) {
        setThreadAffinity(allowedCores);

        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::ostringstream oss;
            oss << "[ReportWriter] Error: Failed to open file " << filename << " for writing.\n";
            if (!logQueue_.push(oss.str())) {
                droppedLogs_.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }

        outfile << "# Report Data\n";
        outfile << "# " << std::left << std::setw(10) << "FrameID" 
                << std::setw(20) << "Timestamp" 
                << std::setw(20) << "ElapsedTime" 
                << std::setw(15) << "NumPoints" 
                << std::setw(15) << "OccMapSize" << "\n";

        ReportDataFrame data;
        int lastReportedDrops = 0;

        while (running_.load(std::memory_order_acquire)) {
            if (reportExtractClusterQueue_.pop(data)) {
                outfile << std::left << std::setw(10) << data.frameID
                        << std::fixed << std::setprecision(6) 
                        << std::setw(20) << data.timestamp
                        << std::setw(20) << data.elapsedTime
                        << std::setw(15) << data.numpoint
                        << std::setw(15) << data.occmapsize << "\n";

                int currentDrops = droppedExtractClusterReports_.load(std::memory_order_relaxed);
                if (currentDrops > lastReportedDrops && (currentDrops - lastReportedDrops) >= 100) {
                    std::ostringstream oss;
                    oss << "[ReportWriter] Warning: " << (currentDrops - lastReportedDrops) 
                        << " cluster extraction report entries dropped due to queue overflow.\n";
                    if (!logQueue_.push(oss.str())) {
                        droppedLogs_.fetch_add(1, std::memory_order_relaxed);
                    }
                    lastReportedDrops = currentDrops;
                }
            } else {
                std::this_thread::yield();
            }
        }

        while (reportExtractClusterQueue_.pop(data)) {
            outfile << std::left << std::setw(10) << data.frameID
                    << std::fixed << std::setprecision(6) 
                    << std::setw(20) << data.timestamp
                    << std::setw(20) << data.elapsedTime
                    << std::setw(15) << data.numpoint
                    << std::setw(15) << data.occmapsize << "\n";
        }

        int finalDrops = droppedExtractClusterReports_.load(std::memory_order_relaxed);
        if (finalDrops > lastReportedDrops) {
            std::ostringstream oss;
            oss << "[ReportWriter] Final report: " << (finalDrops - lastReportedDrops) 
                << " cluster extraction report entries dropped.\n";
            if (!logQueue_.push(oss.str())) {
                droppedLogs_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        outfile.flush();
        outfile.close();
    }

    void Pipeline::setThreadAffinity(const std::vector<int>& coreIDs) {
        if (coreIDs.empty()) {
            if (!logQueue_.push("Warning: [ThreadAffinity] No core IDs provided.\n")) {
                droppedLogs_.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        const unsigned int maxCores = std::thread::hardware_concurrency();
        uint32_t validCores = 0;

        for (int coreID : coreIDs) {
            if (coreID >= 0 && static_cast<unsigned>(coreID) < maxCores) {
                CPU_SET(coreID, &cpuset);
                validCores |= (1 << coreID);
            }
        }

        if (!validCores) {
            if (!logQueue_.push("Error: [ThreadAffinity] No valid core IDs provided.\n")) {
                droppedLogs_.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }

        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
            std::ostringstream oss;
            oss << "Fatal: [ThreadAffinity] Failed to set affinity: " << strerror(errno) << "\n";
            if (!logQueue_.push(oss.str())) {
                droppedLogs_.fetch_add(1, std::memory_order_relaxed);
            }
            running_.store(false, std::memory_order_release);
        }

        std::ostringstream oss;
        oss << "Thread restricted to cores: ";
        for (int coreID : coreIDs) {
            if (CPU_ISSET(coreID, &cpuset)) {
                oss << coreID << " ";
            }
        }
        oss << "\n";
        if (!logQueue_.push(oss.str())) {
            droppedLogs_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void Pipeline::startPointsListener(boost::asio::io_context& ioContext, 
                                      const std::string& host, 
                                      uint16_t port,
                                      uint32_t bufferSize, 
                                      const std::vector<int>& allowedCores) {
        setThreadAffinity(allowedCores);

        if (host.empty() || port == 0) {
            std::ostringstream oss;
            oss << "[PointsListener] Invalid host or port: host='" << host << "', port=" << port << '\n';
            if (!logQueue_.push(oss.str())) {
                droppedLogs_.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }

        std::string hostPortStr = std::string(host) + ":" + std::to_string(port);

        try {
            UDPSocket listener(ioContext, host, port, [&](const std::vector<uint8_t>& data) {
                CallbackPoints::Points decodedPoints;
                callbackPointsProcessor_.process(data, decodedPoints);
                std::cout << "[decodedPoints.numInput]: " << decodedPoints.numInput << "\n";
                if (decodedPoints.frameID != 0 && decodedPoints.numInput > 0) {
                    const Eigen::Vector3d vehiclePosition = decodedPoints.NED;
                    const uint32_t parallelThreshold = 1000;

                    std::vector<Eigen::Vector3d> filteredPt;
                    std::vector<Eigen::Vector3d> filteredAtt;
                    filteredPt.reserve(decodedPoints.numInput);
                    filteredAtt.reserve(decodedPoints.numInput);

                    if (decodedPoints.numInput >= parallelThreshold) {
                        tbb::concurrent_vector<Eigen::Vector3d> tempPt;
                        tbb::concurrent_vector<Eigen::Vector3d> tempAtt;
                        tempPt.reserve(decodedPoints.numInput);
                        tempAtt.reserve(decodedPoints.numInput);

                        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, decodedPoints.numInput),
                            [&](const tbb::blocked_range<uint32_t>& range) {
                                for (uint32_t i = range.begin(); i != range.end(); ++i) {
                                    const Eigen::Vector3d& point = decodedPoints.pt[i];
                                    double distance = (point - vehiclePosition).norm();
                                    if (distance >= processConfig_.mapMinDistance && distance <= processConfig_.mapMaxDistance) {
                                        tempPt.push_back(point);
                                        tempAtt.push_back(decodedPoints.att[i]);
                                    }
                                }
                            });

                        filteredPt = std::vector<Eigen::Vector3d>(tempPt.begin(), tempPt.end());
                        filteredAtt = std::vector<Eigen::Vector3d>(tempAtt.begin(), tempAtt.end());
                    } else {
                        for (uint32_t i = 0; i < decodedPoints.numInput; ++i) {
                            const Eigen::Vector3d& point = decodedPoints.pt[i];
                            double distance = (point - vehiclePosition).norm();
                            if (distance >= processConfig_.mapMinDistance && distance <= processConfig_.mapMaxDistance) {
                                filteredPt.push_back(point);
                                filteredAtt.push_back(decodedPoints.att[i]);
                            }
                        }
                    }

                    decodedPoints.pt = std::move(filteredPt);
                    decodedPoints.att = std::move(filteredAtt);
                    decodedPoints.numInput = static_cast<uint32_t>(filteredPt.size());

                    if (decodedPoints.numInput == 0) return;

                    OccupancyMapDataFrame occMapFrame;
                    ClusterExtractorDataFrame extClsFrame;
                    VehiclePoseDataFrame vehPose;

                    tbb::parallel_invoke(
                        [this, &decodedPoints, &occMapFrame, parallelThreshold]() noexcept {
                            occMapFrame.frameID = decodedPoints.frameID;
                            occMapFrame.timestamp = decodedPoints.t;
                            occMapFrame.vehiclePosition = decodedPoints.NED;
                            occMapFrame.pointcloud.resize(decodedPoints.numInput);

                            if (decodedPoints.numInput >= parallelThreshold) {
                                tbb::parallel_for(tbb::blocked_range<uint32_t>(0, decodedPoints.numInput),
                                    [&occMapFrame, &decodedPoints](const tbb::blocked_range<uint32_t>& range) {
                                        for (uint32_t i = range.begin(); i != range.end(); ++i) {
                                            occMapFrame.pointcloud[i].Pt = decodedPoints.pt[i];
                                            occMapFrame.pointcloud[i].Att = decodedPoints.att[i];
                                        }
                                    });
                            } else {
                                for (uint32_t i = 0; i < decodedPoints.numInput; ++i) {
                                    occMapFrame.pointcloud[i].Pt = decodedPoints.pt[i];
                                    occMapFrame.pointcloud[i].Att = decodedPoints.att[i];
                                }
                            }
                        },
                        [this, &decodedPoints, &extClsFrame, parallelThreshold]() noexcept {
                            extClsFrame.frameID = decodedPoints.frameID;
                            extClsFrame.timestamp = decodedPoints.t;
                            extClsFrame.pointcloud.resize(decodedPoints.numInput);

                            if (decodedPoints.numInput >= parallelThreshold) {
                                tbb::parallel_for(tbb::blocked_range<uint32_t>(0, decodedPoints.numInput),
                                    [&extClsFrame, &decodedPoints](const tbb::blocked_range<uint32_t>& range) {
                                        for (uint32_t i = range.begin(); i != range.end(); ++i) {
                                            extClsFrame.pointcloud[i].Pt = decodedPoints.pt[i];
                                            extClsFrame.pointcloud[i].Att = decodedPoints.att[i];
                                        }
                                    });
                            } else {
                                for (uint32_t i = 0; i < decodedPoints.numInput; ++i) {
                                    extClsFrame.pointcloud[i].Pt = decodedPoints.pt[i];
                                    extClsFrame.pointcloud[i].Att = decodedPoints.att[i];
                                }
                            }
                        },
                        [this, &decodedPoints, &vehPose]() noexcept {
                            vehPose.frameID = decodedPoints.frameID;
                            vehPose.timestamp = decodedPoints.t;
                            vehPose.NED = decodedPoints.NED;
                            vehPose.RPY = decodedPoints.RPY;
                        }
                    );

                    if (!pointsRingBufferOccMap_.push(occMapFrame)) {
                        if (!logQueue_.push("[PointsListener] Ring buffer full for Occ Map; decoded points dropped!\n")) {
                            droppedLogs_.fetch_add(1, std::memory_order_relaxed);
                        }
                    }

                    if (!pointsRingBufferExtCls_.push(extClsFrame)) {
                        if (!logQueue_.push("[PointsListener] Ring buffer full for Ext Cls; decoded points dropped!\n")) {
                            droppedLogs_.fetch_add(1, std::memory_order_relaxed);
                        }
                    }

                    if (!ringBufferPose_.push(vehPose)) {
                        if (!logQueue_.push("[PointsListener] Ring buffer full for Veh Pose; decoded points dropped!\n")) {
                            droppedLogs_.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            }, bufferSize);

            constexpr int maxErrors = 5;
            int errorCount = 0;

            while (running_.load(std::memory_order_acquire)) {
                try {
                    ioContext.run();
                    break;
                } catch (const std::exception& e) {
                    errorCount++;
                    if (errorCount <= maxErrors) {
                        std::ostringstream oss;
                        oss << "[PointsListener] Error: " << e.what() << ". Restarting...\n";
                        if (!logQueue_.push(oss.str())) {
                            droppedLogs_.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    if (errorCount == maxErrors) {
                        if (!logQueue_.push("[PointsListener] Error log limit reached.\n")) {
                            droppedLogs_.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    ioContext.restart();
                }
            }

            ioContext.stop();
            if (!logQueue_.push("[PointsListener] Stopped listener on " + hostPortStr + "\n")) {
                droppedLogs_.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "[PointsListener] Failed to start listener on " + hostPortStr << ": " << e.what() << '\n';
            if (!logQueue_.push(oss.str())) {
                droppedLogs_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    void Pipeline::runOccupancyMapPipeline(const std::vector<int>& allowedCores) {
        setThreadAffinity(allowedCores);

        constexpr auto targetCycleDuration = std::chrono::milliseconds(100);
        std::vector<Voxel3D> occMapVoxels;

        while (running_.load(std::memory_order_acquire)) {
            auto cycleStartTime = std::chrono::steady_clock::now();
            OccupancyMapDataFrame localMapDataFrame;

            size_t itemsToProcess = pointsRingBufferOccMap_.read_available();
            if (itemsToProcess > 0) {
                for (size_t i = 0; i < itemsToProcess; ++i) {
                    if (pointsRingBufferOccMap_.pop(localMapDataFrame)) {
                        // Keep updating with the latest item
                    }
                }
                occupancyMapInstance_->occupancyMap(localMapDataFrame);
                occMapVoxels = occupancyMapInstance_->getOccupiedVoxel();

                if (!voxelsRingBufferOccMap_.push(std::move(occMapVoxels))) {
                    if (!logQueue_.push("[OccupancyMapPipeline] Voxel buffer full; data dropped!\n")) {
                        droppedLogs_.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                ReportDataFrame reportOccupancyMap;
                reportOccupancyMap.frameID = localMapDataFrame.frameID;
                reportOccupancyMap.timestamp = localMapDataFrame.timestamp;
                reportOccupancyMap.numpoint = localMapDataFrame.pointcloud.size();
                reportOccupancyMap.occmapsize = occMapVoxels.size();
                reportOccupancyMap.elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(
                    std::chrono::steady_clock::now() - cycleStartTime).count();

                if (!reportOccupancyMapQueue_.push(reportOccupancyMap)) {  // Corrected to reportOccupancyMapQueue_
                    if (!logQueue_.push("[OccupancyMapPipeline] Report queue full; data dropped!\n")) {
                        droppedLogs_.fetch_add(1, std::memory_order_relaxed);
                    }
                    droppedOccupancyMapReports_.fetch_add(1, std::memory_order_relaxed);
                }
            }

            auto cycleEndTime = std::chrono::steady_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(cycleEndTime - cycleStartTime);
            if (elapsedTime < targetCycleDuration) {
                std::this_thread::sleep_for(targetCycleDuration - elapsedTime);
            } else if (elapsedTime > targetCycleDuration + std::chrono::milliseconds(10)) {
                std::ostringstream oss;
                oss << "Warning: [OccupancyMapPipeline] Processing exceeded target by " << (elapsedTime - targetCycleDuration).count() << " ms\n";
                if (!logQueue_.push(oss.str())) {
                    droppedLogs_.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    void Pipeline::runClusterExtractionPipeline(const std::vector<int>& allowedCores) {
        setThreadAffinity(allowedCores);

        constexpr auto targetCycleDuration = std::chrono::milliseconds(100);
        std::vector<Voxel3D> extClsVoxels;

        while (running_.load(std::memory_order_acquire)) {
            auto cycleStartTime = std::chrono::steady_clock::now();
            ClusterExtractorDataFrame localExtractorDataFrame;

            size_t itemsToProcess = pointsRingBufferExtCls_.read_available();
            if (itemsToProcess > 0) {
                for (size_t i = 0; i < itemsToProcess; ++i) {
                    if (pointsRingBufferExtCls_.pop(localExtractorDataFrame)) {
                        // Process each frame
                    }
                }
                clusterExtractionInstance_->extractClusters(localExtractorDataFrame);
                extClsVoxels = clusterExtractionInstance_->getOccupiedVoxel();

                if (!voxelsRingBufferExtCls_.push(std::move(extClsVoxels))) {
                    if (!logQueue_.push("[ClusterExtractionPipeline] Voxel buffer full; data dropped!\n")) {
                        droppedLogs_.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                ReportDataFrame reportClusterExtractor;
                reportClusterExtractor.frameID = localExtractorDataFrame.frameID;
                reportClusterExtractor.timestamp = localExtractorDataFrame.timestamp;
                reportClusterExtractor.numpoint = localExtractorDataFrame.pointcloud.size();
                reportClusterExtractor.occmapsize = extClsVoxels.size();
                reportClusterExtractor.elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(
                    std::chrono::steady_clock::now() - cycleStartTime).count();

                if (!reportExtractClusterQueue_.push(reportClusterExtractor)) {
                    if (!logQueue_.push("[ClusterExtractionPipeline] Report queue full; data dropped!\n")) {
                        droppedLogs_.fetch_add(1, std::memory_order_relaxed);
                    }
                    droppedExtractClusterReports_.fetch_add(1, std::memory_order_relaxed);
                }
            }

            auto cycleEndTime = std::chrono::steady_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(cycleEndTime - cycleStartTime);

            if (elapsedTime < targetCycleDuration) {
                std::this_thread::sleep_for(targetCycleDuration - elapsedTime);
            } else if (elapsedTime > targetCycleDuration + std::chrono::milliseconds(10)) {
                std::ostringstream oss;
                oss << "Warning: [ClusterExtractionPipeline] Processing exceeded target by " 
                    << (elapsedTime - targetCycleDuration).count() << " ms\n";
                if (!logQueue_.push(oss.str())) {
                    droppedLogs_.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    void Pipeline::runVizualizationPipeline(const std::vector<int>& allowedCores) {
        setThreadAffinity(allowedCores);

        open3d::visualization::Visualizer vis;
        vis.CreateVisualizerWindow("3D Voxel Visualization - Ocean View", 1280, 720);
        vis.GetRenderOption().background_color_ = Eigen::Vector3d(0.0, 0.2, 0.5);

        vis.AddGeometry(voxel_grid_occMap_ptr_);
        vis.AddGeometry(voxel_grid_extCls_ptr_);
        vis.AddGeometry(vehicle_mesh_ptr_);

        auto& view = vis.GetViewControl();
        view.SetFront({0, -1, -1});
        view.SetUp({0, 1, 0});
        view.SetLookat({0, 0, 0});
        view.SetZoom(0.5);

        vis.RegisterAnimationCallback([&](open3d::visualization::Visualizer* vis_ptr) {
            return updateVisualization(vis_ptr);
        });

        vis.Run();
        vis.DestroyVisualizerWindow();
    }

    bool Pipeline::updateVisualization(open3d::visualization::Visualizer* vis) {
        bool updated = false;

        size_t itemsToProcessVoxelOccMap = voxelsRingBufferOccMap_.read_available();
        if (itemsToProcessVoxelOccMap > 0) {
            std::vector<Voxel3D> localVoxelProcessOccMap;
            for (size_t i = 0; i < itemsToProcessVoxelOccMap; ++i) {
                if (voxelsRingBufferOccMap_.pop(localVoxelProcessOccMap)) {
                    // Keep the latest data
                }
            }
            if (!localVoxelProcessOccMap.empty()) {
                voxel_grid_occMap_ptr_->voxels_ = createVoxelGrid(localVoxelProcessOccMap, mapConfig_.mapOrigin, mapConfig_.resolution)->voxels_;
                vis->UpdateGeometry(voxel_grid_occMap_ptr_);
                updated = true;
            }
        }

        size_t itemsToProcessVoxelExtCls = voxelsRingBufferExtCls_.read_available();
        if (itemsToProcessVoxelExtCls > 0) {
            std::vector<Voxel3D> localVoxelProcessExtCls;
            for (size_t i = 0; i < itemsToProcessVoxelExtCls; ++i) {
                if (voxelsRingBufferExtCls_.pop(localVoxelProcessExtCls)) {
                    // Keep the latest data
                }
            }
            if (!localVoxelProcessExtCls.empty()) {
                voxel_grid_extCls_ptr_->voxels_ = createVoxelGrid(localVoxelProcessExtCls, mapConfig_.mapOrigin, mapConfig_.resolution)->voxels_;
                vis->UpdateGeometry(voxel_grid_extCls_ptr_);
                updated = true;
            }
        }

        size_t itemsToProcessVehPose = ringBufferPose_.read_available();
        if (itemsToProcessVehPose > 0) {
            VehiclePoseDataFrame localProcessVehPose;
            for (size_t i = 0; i < itemsToProcessVehPose; ++i) {
                if (ringBufferPose_.pop(localProcessVehPose)) {
                    // Keep the latest pose
                }
            }
            if (vehicle_mesh_ptr_) {
                auto new_vehicle_mesh = createVehicleMesh(localProcessVehPose.NED, localProcessVehPose.RPY);
                vehicle_mesh_ptr_->vertices_ = new_vehicle_mesh->vertices_;
                vehicle_mesh_ptr_->triangles_ = new_vehicle_mesh->triangles_;
                vehicle_mesh_ptr_->vertex_colors_ = new_vehicle_mesh->vertex_colors_;
                vis->UpdateGeometry(vehicle_mesh_ptr_);
                updated = true;
            }
        }

        return running_.load(std::memory_order_acquire) || updated;
    }
}  // namespace slam