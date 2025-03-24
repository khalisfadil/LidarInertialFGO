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

namespace slam {

    // -----------------------------------------------------------------------------
    // Section: assignVoxelColorsRed
    // -----------------------------------------------------------------------------
    boost::lockfree::spsc_queue<VehiclePoseDataFrame, boost::lockfree::capacity<128>> Pipeline::ringBufferPose;
    boost::lockfree::spsc_queue<OccupancyMapDataFrame, boost::lockfree::capacity<128>> Pipeline::pointsRingBufferOccMap;
    boost::lockfree::spsc_queue<std::vector<Voxel3D>, boost::lockfree::capacity<128>> Pipeline::voxelsRingBufferOccMap;
    boost::lockfree::spsc_queue<ClusterExtractorDataFrame, boost::lockfree::capacity<128>> Pipeline::pointsRingBufferExtCls;
    boost::lockfree::spsc_queue<std::vector<Voxel3D>, boost::lockfree::capacity<128>> Pipeline::voxelsRingBufferExtCls;
    
    boost::lockfree::spsc_queue<std::string, boost::lockfree::capacity<1024>> Pipeline::logQueue;
    boost::lockfree::spsc_queue<ReportDataFrame, boost::lockfree::capacity<1024>> Pipeline::reportOccupancyMapQueue;
    boost::lockfree::spsc_queue<ReportDataFrame, boost::lockfree::capacity<1024>> Pipeline::reportExtractClusterQueue;

    std::atomic<bool> Pipeline::running{true};
    std::atomic<int> Pipeline::droppedLogs{0};
    std::atomic<int> Pipeline::droppedOccupancyMapReports{0};
    std::atomic<int> Pipeline::droppedExtractClusterReports{0};
    std::condition_variable Pipeline::globalCV;
    std::thread Pipeline::logThread_; // Static definition

    std::shared_ptr<open3d::geometry::VoxelGrid> Pipeline::voxel_grid_occMap_ptr;
    std::shared_ptr<open3d::geometry::VoxelGrid> Pipeline::voxel_grid_extCls_ptr;
    std::shared_ptr<open3d::geometry::TriangleMesh> Pipeline::vehicle_mesh_ptr;

    // -----------------------------------------------------------------------------
    // Section: assignVoxelColorsRed
    // -----------------------------------------------------------------------------

    Pipeline& Pipeline::getInstance() noexcept {
        static Pipeline instance;
        return instance;
    }

    // -----------------------------------------------------------------------------
    // Section: assignVoxelColorsRed
    // -----------------------------------------------------------------------------

    Pipeline::Pipeline() {
        if (!occupancyMapInstance) {
            occupancyMapInstance = std::make_unique<occmap::OccupancyMap>(
                mapConfig_.resolution, mapConfig_.mapMaxDistance, mapConfig_.mapOrigin,
                mapConfig_.maxPointsPerVoxel, mapConfig_.colorMode);
        }

        if (!clusterExtractionInstance) {
            clusterExtractionInstance = std::make_unique<cluster::ClusterExtraction>(
                mapConfig_.resolution, mapConfig_.mapOrigin, mapConfig_.tolerance, mapConfig_.min_size, mapConfig_.max_size,
                mapConfig_.max_frames, mapConfig_.maxPointsPerVoxel, mapConfig_.colorMode);
        }

    }

    // -----------------------------------------------------------------------------
    // Section: processLogQueue
    // -----------------------------------------------------------------------------

    void Pipeline::processLogQueue(const std::vector<int>& allowedCores) noexcept {
        setThreadAffinity(allowedCores); // Pin logging thread to specified cores

        std::string message;
        int lastReportedDrops = 0;
        while (running.load(std::memory_order_acquire)) {
            if (logQueue.pop(message)) {
                std::cerr << message;
                int currentDrops = droppedLogs.load(std::memory_order_relaxed);
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
        while (logQueue.pop(message)) {
            std::cerr << message;
        }
        int finalDrops = droppedLogs.load(std::memory_order_relaxed);
        if (finalDrops > lastReportedDrops) {
            std::cerr << "[Logging] Final report: " << (finalDrops - lastReportedDrops) << " log messages dropped.\n";
        }
    }

    // -----------------------------------------------------------------------------
    // Section: processLogQueue
    // -----------------------------------------------------------------------------

    void Pipeline::processReportQueueOccMap(const std::string& filename, const std::vector<int>& allowedCores) noexcept {
        setThreadAffinity(allowedCores);

        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::ostringstream oss;
            oss << "[ReportWriter] Error: Failed to open file " << filename << " for writing.\n";
            if (!logQueue.push(oss.str())) {
                droppedLogs.fetch_add(1, std::memory_order_relaxed);
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

        while (running.load(std::memory_order_acquire)) {
            if (reportOccupancyMapQueue.pop(data)) {
                outfile << std::left << std::setw(10) << data.frameID
                        << std::fixed << std::setprecision(6) 
                        << std::setw(20) << data.timestamp
                        << std::setw(20) << data.elapsedTime
                        << std::setw(15) << data.numpoint
                        << std::setw(15) << data.occmapsize << "\n";

                int currentDrops = droppedOccupancyMapReports.load(std::memory_order_relaxed);
                if (currentDrops > lastReportedDrops && (currentDrops - lastReportedDrops) >= 100) {
                    std::ostringstream oss;
                    oss << "[ReportWriter] Warning: " << (currentDrops - lastReportedDrops) 
                        << " occupancy map report entries dropped due to queue overflow.\n";
                    if (!logQueue.push(oss.str())) {
                        droppedLogs.fetch_add(1, std::memory_order_relaxed);
                    }
                    lastReportedDrops = currentDrops;
                }
            } else {
                std::this_thread::yield();
            }
        }

        while (reportOccupancyMapQueue.pop(data)) {
            outfile << std::left << std::setw(10) << data.frameID
                    << std::fixed << std::setprecision(6) 
                    << std::setw(20) << data.timestamp
                    << std::setw(20) << data.elapsedTime
                    << std::setw(15) << data.numpoint
                    << std::setw(15) << data.occmapsize << "\n";
        }

        int finalDrops = droppedOccupancyMapReports.load(std::memory_order_relaxed);
        if (finalDrops > lastReportedDrops) {
            std::ostringstream oss;
            oss << "[ReportWriter] Final report: " << (finalDrops - lastReportedDrops) 
                << " occupancy map report entries dropped.\n";
            if (!logQueue.push(oss.str())) {
                droppedLogs.fetch_add(1, std::memory_order_relaxed);
            }
        }

        outfile.flush(); // Ensure data is written
        outfile.close();
    }

    // -----------------------------------------------------------------------------
    // Section: processLogQueue
    // -----------------------------------------------------------------------------

    void Pipeline::processReportQueueExtCls(const std::string& filename, const std::vector<int>& allowedCores) noexcept {
        setThreadAffinity(allowedCores);

        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::ostringstream oss;
            oss << "[ReportWriter] Error: Failed to open file " << filename << " for writing.\n";
            if (!logQueue.push(oss.str())) {
                droppedLogs.fetch_add(1, std::memory_order_relaxed);
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

        while (running.load(std::memory_order_acquire)) {
            if (reportExtractClusterQueue.pop(data)) {
                outfile << std::left << std::setw(10) << data.frameID
                        << std::fixed << std::setprecision(6) 
                        << std::setw(20) << data.timestamp
                        << std::setw(20) << data.elapsedTime
                        << std::setw(15) << data.numpoint
                        << std::setw(15) << data.occmapsize << "\n";

                int currentDrops = droppedExtractClusterReports.load(std::memory_order_relaxed);
                if (currentDrops > lastReportedDrops && (currentDrops - lastReportedDrops) >= 100) {
                    std::ostringstream oss;
                    oss << "[ReportWriter] Warning: " << (currentDrops - lastReportedDrops) 
                        << " cluster extraction report entries dropped due to queue overflow.\n";
                    if (!logQueue.push(oss.str())) {
                        droppedLogs.fetch_add(1, std::memory_order_relaxed);
                    }
                    lastReportedDrops = currentDrops;
                }
            } else {
                std::this_thread::yield();
            }
        }

        while (reportExtractClusterQueue.pop(data)) {
            outfile << std::left << std::setw(10) << data.frameID
                    << std::fixed << std::setprecision(6) 
                    << std::setw(20) << data.timestamp
                    << std::setw(20) << data.elapsedTime
                    << std::setw(15) << data.numpoint
                    << std::setw(15) << data.occmapsize << "\n";
        }

        int finalDrops = droppedExtractClusterReports.load(std::memory_order_relaxed);
        if (finalDrops > lastReportedDrops) {
            std::ostringstream oss;
            oss << "[ReportWriter] Final report: " << (finalDrops - lastReportedDrops) 
                << " cluster extraction report entries dropped.\n";
            if (!logQueue.push(oss.str())) {
                droppedLogs.fetch_add(1, std::memory_order_relaxed);
            }
        }

        outfile.flush(); // Ensure data is written
        outfile.close();
    }

    // -----------------------------------------------------------------------------
    // Section: signalHandler
    // -----------------------------------------------------------------------------

    void Pipeline::signalHandler(int signal) noexcept {
        if (signal == SIGINT || signal == SIGTERM) {
            running.store(false, std::memory_order_release);
            globalCV.notify_all();

            constexpr const char* message = "[signalHandler] Shutting down...\n";
            constexpr size_t messageLen = sizeof(message) - 1;
            ssize_t result = write(STDOUT_FILENO, message, messageLen);
            if (result == -1) {
                // Handle error (e.g., silently ignore or log elsewhere if possible)
                // Note: Avoid complex operations in signal handlers
            }
        }
    }

    // -----------------------------------------------------------------------------
    // Section: assignVoxelColorsRed
    // -----------------------------------------------------------------------------

    void Pipeline::setThreadAffinity(const std::vector<int>& coreIDs) noexcept {
        if (coreIDs.empty()) {
            if (!logQueue.push("Warning: [ThreadAffinity] No core IDs provided.\n")) {
                droppedLogs.fetch_add(1, std::memory_order_relaxed);
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
            if (!logQueue.push("Error: [ThreadAffinity] No valid core IDs provided.\n")) {
                droppedLogs.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }

        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
            std::ostringstream oss;
            oss << "Fatal: [ThreadAffinity] Failed to set affinity: " << strerror(errno) << "\n";
            if (!logQueue.push(oss.str())) {
                droppedLogs.fetch_add(1, std::memory_order_relaxed);
            }
            running.store(false); // Optionally terminate
        }

        std::ostringstream oss;
        oss << "Thread restricted to cores: ";
        for (int coreID : coreIDs) {
            if (CPU_ISSET(coreID, &cpuset)) {
                oss << coreID << " ";
            }
        }
        oss << "\n";
        if (!logQueue.push(oss.str())) {
            droppedLogs.fetch_add(1, std::memory_order_relaxed);
        }

    }

    // -----------------------------------------------------------------------------
    // Section: assignVoxelColorsRed
    // -----------------------------------------------------------------------------

    void Pipeline::startPointsListener(boost::asio::io_context& ioContext,
                                        std::string_view host,
                                        uint16_t port,
                                        uint32_t bufferSize,
                                        const std::vector<int>& allowedCores) noexcept {
        setThreadAffinity(allowedCores);

        if (host.empty() || port == 0) {
            std::ostringstream oss;
            oss << "[PointsListener] Invalid host or port: host='" << host << "', port=" << port << '\n';
            if (!logQueue.push(oss.str())) {
                droppedLogs.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }

        std::string hostPortStr = std::string(host) + ":" + std::to_string(port);

        try {
            UDPSocket listener(ioContext, std::string(host), port,
                [this](const std::vector<uint8_t>& data) noexcept {
                    CallbackPoints::Points decodedPoints;

                    callbackPointsProcessor.process(data, decodedPoints);

                    if (decodedPoints.frameID != 0 && decodedPoints.numInput > 0) {
                        const Eigen::Vector3d vehiclePosition = decodedPoints.NED;
                        const uint32_t parallelThreshold = 1000; // Define threshold here, can be adjusted

                        // Use regular vectors since we'll handle parallel/serial separately
                        std::vector<Eigen::Vector3d> filteredPt;
                        std::vector<Eigen::Vector3d> filteredAtt;
                        
                        filteredPt.reserve(decodedPoints.numInput);
                        filteredAtt.reserve(decodedPoints.numInput);

                        // Filter points with threshold-based parallel/serial execution
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

                        // Update decodedPoints
                        decodedPoints.pt = std::move(filteredPt);
                        decodedPoints.att = std::move(filteredAtt);
                        decodedPoints.numInput = static_cast<uint32_t>(filteredPt.size());

                        if (decodedPoints.numInput == 0) return;

                        // Temporary storage for transformed data
                        OccupancyMapDataFrame occMapFrame;
                        ClusterExtractorDataFrame extClsFrame;
                        VehiclePoseDataFrame vehPose;

                        // Parallel transformation into the new structures
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

                        // Push to OccupancyMap ring buffer
                        if (!pointsRingBufferOccMap.push(occMapFrame)) {
                            if (!logQueue.push("[PointsListener] Ring buffer full for Occ Map; decoded points dropped!\n")) {
                                droppedLogs.fetch_add(1, std::memory_order_relaxed);
                            }
                        }

                        // Push to ClusterExtractor ring buffer
                        if (!pointsRingBufferExtCls.push(extClsFrame)) {
                            if (!logQueue.push("[PointsListener] Ring buffer full for Ext Cls; decoded points dropped!\n")) {
                                droppedLogs.fetch_add(1, std::memory_order_relaxed);
                            }
                        }

                        // Push to VehiclePose ring buffer
                        if (!ringBufferPose.push(vehPose)) {
                            if (!logQueue.push("[PointsListener] Ring buffer full for Veh Pose; decoded points dropped!\n")) {
                                droppedLogs.fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                    }
                }, bufferSize);

            listener.setReceiveBufferSize(bufferSize);

            constexpr int maxErrors = 5;
            int errorCount = 0;

            while (running.load(std::memory_order_acquire)) {
                try {
                    ioContext.run();
                    break;
                } catch (const std::exception& e) {
                    errorCount++;
                    if (errorCount <= maxErrors) {
                        std::ostringstream oss;
                        oss << "[PointsListener] Error: " << e.what() << ". Restarting...\n";
                        if (!logQueue.push(oss.str())) {
                            droppedLogs.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    if (errorCount == maxErrors) {
                        if (!logQueue.push("[PointsListener] Error log limit reached.\n")) {
                            droppedLogs.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    ioContext.restart();
                }
            }

            listener.stop();
            if (!logQueue.push("[PointsListener] Stopped listener on " + hostPortStr + "\n")) {
                droppedLogs.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "[PointsListener] Failed to start listener on " + hostPortStr << ": " << e.what() << '\n';
            if (!logQueue.push(oss.str())) {
                droppedLogs.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // -----------------------------------------------------------------------------
    // Section: assignVoxelColorsRed
    // -----------------------------------------------------------------------------

    void Pipeline::runOccupancyMapPipeline(const std::vector<int>& allowedCores) noexcept {
        setThreadAffinity(allowedCores);

        constexpr auto targetCycleDuration = std::chrono::milliseconds(100);

        // Pre-allocate voxel buffer to reduce allocation overhead
        std::vector<Voxel3D> occMapVoxels;

        while (running.load(std::memory_order_acquire)) {
            auto cycleStartTime = std::chrono::steady_clock::now();
            OccupancyMapDataFrame localMapDataFrame;

            size_t itemsToProcess = pointsRingBufferOccMap.read_available();
            if (itemsToProcess > 0) {
                
                for (size_t i = 0; i < itemsToProcess; ++i) {
                    if (pointsRingBufferOccMap.pop(localMapDataFrame)) {
                        // Keep updating localPoints with the latest current item
                    }
                }
                occupancyMapInstance->occupancyMap(localMapDataFrame);
                occMapVoxels = occupancyMapInstance->getOccupiedVoxel();

                if (!voxelsRingBufferOccMap.push(std::move(occMapVoxels))) {
                    if (!logQueue.push("[OccupancyMapPipeline] Voxel buffer full; data dropped!\n")) {
                        droppedLogs.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                // Create and populate report
                ReportDataFrame reportOccupancyMap;
                reportOccupancyMap.frameID = localMapDataFrame.frameID;
                reportOccupancyMap.timestamp = localMapDataFrame.timestamp;
                reportOccupancyMap.numpoint = localMapDataFrame.pointcloud.size();
                reportOccupancyMap.occmapsize = occMapVoxels.size();
                reportOccupancyMap.elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>( // Convert to seconds as double
                    std::chrono::steady_clock::now() - cycleStartTime).count();

                // Push report to the queue
                if (!reportExtractClusterQueue.push(reportOccupancyMap)) {
                    if (!logQueue.push("[OccupancyMapPipeline] Report queue full; data dropped!\n")) {
                        droppedLogs.fetch_add(1, std::memory_order_relaxed);
                    }
                    droppedOccupancyMapReports.fetch_add(1, std::memory_order_relaxed); // Use specific counter
                }
            }

            auto cycleEndTime = std::chrono::steady_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(cycleEndTime - cycleStartTime);
            if (elapsedTime < targetCycleDuration) {
                std::this_thread::sleep_for(targetCycleDuration - elapsedTime);
            } else if (elapsedTime > targetCycleDuration + std::chrono::milliseconds(10)) {
                std::ostringstream oss;
                oss << "Warning: [OccupancyMapPipeline] Processing exceeded target by " << (elapsedTime - targetCycleDuration).count() << " ms\n";
                if (!logQueue.push(oss.str())) {
                    droppedLogs.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    // -----------------------------------------------------------------------------
    // Section: assignVoxelColorsRed
    // -----------------------------------------------------------------------------

    void Pipeline::runClusterExtractionPipeline(const std::vector<int>& allowedCores) noexcept {                       
        setThreadAffinity(allowedCores);

        constexpr auto targetCycleDuration = std::chrono::milliseconds(100);

        // Pre-allocate voxel buffer to reduce allocation overhead
        std::vector<Voxel3D> ExtClsVoxels;

        while (running.load(std::memory_order_acquire)) {
            auto cycleStartTime = std::chrono::steady_clock::now();
            ClusterExtractorDataFrame localExtractorDataFrame;

            size_t itemsToProcess = pointsRingBufferExtCls.read_available();
            if (itemsToProcess > 0) {
                for (size_t i = 0; i < itemsToProcess; ++i) {
                    if (pointsRingBufferExtCls.pop(localExtractorDataFrame)) {
                        // Process each frame (assuming this is intended)
                        // If you want to accumulate points, modify this logic
                    }
                }
                // Extract clusters from the last popped frame
                clusterExtractionInstance->extractClusters(localExtractorDataFrame);
                ExtClsVoxels = clusterExtractionInstance->getOccupiedVoxel();

                // Push voxels to the ring buffer
                if (!voxelsRingBufferExtCls.push(std::move(ExtClsVoxels))) {
                    if (!logQueue.push("[ClusterExtractionPipeline] Voxel buffer full; data dropped!\n")) {
                        droppedLogs.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                // Create and populate report
                ReportDataFrame reportClusterExtractor;
                reportClusterExtractor.frameID = localExtractorDataFrame.frameID;
                reportClusterExtractor.timestamp = localExtractorDataFrame.timestamp;
                reportClusterExtractor.numpoint = localExtractorDataFrame.pointcloud.size();
                reportClusterExtractor.occmapsize = ExtClsVoxels.size();
                reportClusterExtractor.elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>( // Convert to seconds as double
                    std::chrono::steady_clock::now() - cycleStartTime).count();

                // Push report to the queue
                if (!reportExtractClusterQueue.push(reportClusterExtractor)) {
                    if (!logQueue.push("[ClusterExtractionPipeline] Report queue full; data dropped!\n")) {
                        droppedLogs.fetch_add(1, std::memory_order_relaxed);
                    }
                    droppedExtractClusterReports.fetch_add(1, std::memory_order_relaxed); // Use specific counter
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
                if (!logQueue.push(oss.str())) {
                    droppedLogs.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    // -----------------------------------------------------------------------------
    // Section: assignVoxelColorsRed
    // -----------------------------------------------------------------------------

    void Pipeline::runVizualizationPipeline(const std::vector<int>& allowedCores) noexcept {  

        setThreadAffinity(allowedCores);

        open3d::visualization::Visualizer vis;
        vis.CreateVisualizerWindow("3D Voxel Visualization - Ocean View", 1280, 720);
        vis.GetRenderOption().background_color_ = Eigen::Vector3d(0.0, 0.2, 0.5);

        vis.AddGeometry(voxel_grid_occMap_ptr);
        vis.AddGeometry(voxel_grid_extCls_ptr);
        vis.AddGeometry(vehicle_mesh_ptr);

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

    // -----------------------------------------------------------------------------
    // Section: assignVoxelColorsRed
    // -----------------------------------------------------------------------------

    bool Pipeline::updateVisualization(open3d::visualization::Visualizer* vis) noexcept {
        bool updated = false;

        // Process Occupancy Map Voxels
        size_t itemsToProcessVoxelOccMap = voxelsRingBufferOccMap.read_available();
        if (itemsToProcessVoxelOccMap > 0) {
            std::vector<Voxel3D> localVoxelProcessOccMap;
            for (size_t i = 0; i < itemsToProcessVoxelOccMap; ++i) {
                if (voxelsRingBufferOccMap.pop(localVoxelProcessOccMap)) {
                    // Keep the latest data; could accumulate if needed
                }
            }
            if (!localVoxelProcessOccMap.empty()) {
                voxel_grid_occMap_ptr->voxels_ = createVoxelGrid(localVoxelProcessOccMap, mapConfig_.mapOrigin, mapConfig_.resolution)->voxels_;
                vis->UpdateGeometry(voxel_grid_occMap_ptr);
                updated = true;
            }
        }

        // Process Cluster Extraction Voxels
        size_t itemsToProcessVoxelExtCls = voxelsRingBufferExtCls.read_available();
        if (itemsToProcessVoxelExtCls > 0) {
            std::vector<Voxel3D> localVoxelProcessExtCls;
            for (size_t i = 0; i < itemsToProcessVoxelExtCls; ++i) {
                if (voxelsRingBufferExtCls.pop(localVoxelProcessExtCls)) { // Corrected from OccMap to ExtCls
                    // Keep the latest data
                }
            }
            if (!localVoxelProcessExtCls.empty()) {
                voxel_grid_extCls_ptr->voxels_ = createVoxelGrid(localVoxelProcessExtCls, mapConfig_.mapOrigin, mapConfig_.resolution)->voxels_;
                vis->UpdateGeometry(voxel_grid_extCls_ptr);
                updated = true;
            }
        }

        // Process Vehicle Pose
        size_t itemsToProcessVehPose = ringBufferPose.read_available();
        if (itemsToProcessVehPose > 0) {
            VehiclePoseDataFrame localProcessVehPose;
            for (size_t i = 0; i < itemsToProcessVehPose; ++i) {
                if (ringBufferPose.pop(localProcessVehPose)) {
                    // Keep the latest pose
                }
            }
            if (vehicle_mesh_ptr) {
                auto new_vehicle_mesh = createVehicleMesh(localProcessVehPose.NED, localProcessVehPose.RPY);
                vehicle_mesh_ptr->vertices_ = new_vehicle_mesh->vertices_;
                vehicle_mesh_ptr->triangles_ = new_vehicle_mesh->triangles_;
                vehicle_mesh_ptr->vertex_colors_ = new_vehicle_mesh->vertex_colors_;
                vis->UpdateGeometry(vehicle_mesh_ptr);
                updated = true;
            }
        }

        return running.load(std::memory_order_acquire) || updated; // Continue if running or updated
    }

}  // namespace slam