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

    std::unique_ptr<occmap::OccupancyMap>  Pipeline::occupancyMapInstance = nullptr;
    std::unique_ptr<cluster::ClusterExtraction>  Pipeline::clusterExtractionInstance = nullptr;

    boost::lockfree::spsc_queue<CallbackPoints::Points, boost::lockfree::capacity<128>> Pipeline::ringBufferPose;
    boost::lockfree::spsc_queue<CallbackPoints::Points, boost::lockfree::capacity<128>> Pipeline::pointsRingBufferOccMap;
    boost::lockfree::spsc_queue<std::vector<Voxel3D>, boost::lockfree::capacity<128>> Pipeline::voxelsRingBufferOccMap;
    boost::lockfree::spsc_queue<CallbackPoints::Points, boost::lockfree::capacity<128>> Pipeline::pointsRingBufferExtCls;
    boost::lockfree::spsc_queue<std::vector<Voxel3D>, boost::lockfree::capacity<128>> Pipeline::voxelsRingBufferExtClsNonPersistent;
    boost::lockfree::spsc_queue<std::vector<Voxel3D>, boost::lockfree::capacity<128>> Pipeline::voxelsRingBufferExtClsPersistent;

    boost::lockfree::spsc_queue<std::string, boost::lockfree::capacity<128>> Pipeline::logQueue;
    boost::lockfree::spsc_queue<ReportDataFrame, boost::lockfree::capacity<128>> Pipeline::reportOccupancyMapQueue;
    boost::lockfree::spsc_queue<ReportDataFrame, boost::lockfree::capacity<128>> Pipeline::reportExtractClusterQueue;

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

    // Pipeline& Pipeline::getInstance() noexcept {
    //     static Pipeline instance;
    //     return instance;
    // }

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
        if (!voxel_grid_occMap_ptr) {
            voxel_grid_occMap_ptr = std::make_shared<open3d::geometry::VoxelGrid>();
            voxel_grid_occMap_ptr->origin_ = mapConfig_.mapOrigin; // e.g., {0,0,0}
            voxel_grid_occMap_ptr->voxel_size_ = mapConfig_.resolution; // e.g., 10
        }
        if (!voxel_grid_extCls_ptr) {
            voxel_grid_extCls_ptr = std::make_shared<open3d::geometry::VoxelGrid>();
            voxel_grid_extCls_ptr->origin_ = mapConfig_.mapOrigin;
            voxel_grid_extCls_ptr->voxel_size_ = mapConfig_.resolution;
        }
        if (!vehicle_mesh_ptr) {
            vehicle_mesh_ptr = std::make_shared<open3d::geometry::TriangleMesh>();
        }

    }

    // -----------------------------------------------------------------------------
    // Section: processLogQueue
    // -----------------------------------------------------------------------------

    void Pipeline::processLogQueue(const std::vector<int>& allowedCores) {
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

    void Pipeline::processReportQueueOccMap(const std::string& filename, const std::vector<int>& allowedCores) {
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

    void Pipeline::processReportQueueExtCls(const std::string& filename, const std::vector<int>& allowedCores) {
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

    void Pipeline::signalHandler(int signal) {
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

    void Pipeline::setThreadAffinity(const std::vector<int>& coreIDs) {
        if (coreIDs.empty()) {
            std::lock_guard<std::mutex> consoleLock(consoleMutex);
            // if (!logQueue.push("Warning: [ThreadAffinity] No core IDs provided.\n")) {
            //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
            // }
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
            // if (!logQueue.push("Error: [ThreadAffinity] No valid core IDs provided.\n")) {
            //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
            // }
            return;
        }

        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
            // std::ostringstream oss;
            // oss << "Fatal: [ThreadAffinity] Failed to set affinity: " << strerror(errno) << "\n";
            // if (!logQueue.push(oss.str())) {
            //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
            // }
            running.store(false); // Optionally terminate
        }

        // std::ostringstream oss;
        // oss << "Thread restricted to cores: ";
        // for (int coreID : coreIDs) {
        //     if (CPU_ISSET(coreID, &cpuset)) {
        //         oss << coreID << " ";
        //     }
        // }
        // oss << "\n";
        // if (!logQueue.push(oss.str())) {
        //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
        // }

    }

    // -----------------------------------------------------------------------------
    // Section: assignVoxelColorsRed
    // -----------------------------------------------------------------------------

    void Pipeline::startPointsListener(boost::asio::io_context& ioContext, 
                                            const std::string& host, 
                                            uint16_t port,
                                            uint32_t bufferSize, 
                                            const std::vector<int>& allowedCores) {
        setThreadAffinity(allowedCores);

        if (host.empty() || port == 0) {
            std::lock_guard<std::mutex> consoleLock(consoleMutex);
            // std::ostringstream oss;
            // oss << "[PointsListener] Invalid host or port: host='" << host << "', port=" << port << '\n';
            // if (!logQueue.push(oss.str())) {
            //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
            // }
            return;
        }

        // std::string hostPortStr = std::string(host) + ":" + std::to_string(port);

        try {
            UDPSocket listener(ioContext, host, port, [&](const std::vector<uint8_t>& data) {

                    CallbackPoints::Points decodedPoints;

                    callbackPointsProcessor.process(data, decodedPoints);

                    if (decodedPoints.frameID != 0) {

                        // {
                        //     std::lock_guard<std::mutex> consoleLock(consoleMutex);  
                        //     std::cerr << "storedDecodedPoints.numInput: " << decodedPoints.numInput << std::endl;
                        // }
                        
                        if (decodedPoints.numInput == 0) return;

                        // Push to OccupancyMap ring buffer
                        if (!pointsRingBufferOccMap.push(decodedPoints)) {
                            // if (!logQueue.push("[PointsListener] Ring buffer full for Occ Map; decoded points dropped!\n")) {
                            //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
                            // }
                        }

                        // if (!pointsRingBufferExtCls.push(decodedPoints)) {
                        //     // if (!logQueue.push("[PointsListener] Ring buffer full for Occ Map; decoded points dropped!\n")) {
                        //     //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
                        //     // }
                        // }

                        // Push to OccupancyMap ring buffer
                        if (!ringBufferPose.push(decodedPoints)) {
                            // if (!logQueue.push("[PointsListener] Ring buffer full for Occ Map; decoded points dropped!\n")) {
                            //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
                            // }
                        }

                    }
                }, bufferSize);

            constexpr int maxErrors = 5;
            int errorCount = 0;

            while (running.load(std::memory_order_acquire)) {
                try {
                    ioContext.run();
                    break;
                } catch (const std::exception& e) {
                    errorCount++;
                    if (errorCount <= maxErrors) {
                        // std::ostringstream oss;
                        // oss << "[PointsListener] Error: " << e.what() << ". Restarting...\n";
                        // if (!logQueue.push(oss.str())) {
                        //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
                        // }
                    }
                    if (errorCount == maxErrors) {
                        // if (!logQueue.push("[PointsListener] Error log limit reached.\n")) {
                        //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
                        // }
                    }
                    ioContext.restart();
                }
            }

            // if (!logQueue.push("[PointsListener] Stopped listener on " + hostPortStr + "\n")) {
            //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
            // }
        } catch (const std::exception& e) {
            // std::ostringstream oss;
            // oss << "[PointsListener] Failed to start listener on " + hostPortStr << ": " << e.what() << '\n';
            // if (!logQueue.push(oss.str())) {
            //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
            // }
        }
        ioContext.stop();
    }

    // -----------------------------------------------------------------------------
    // Section: processPoints
    // -----------------------------------------------------------------------------

    void Pipeline::processPoints(CallbackPoints::Points& decodedPoints){

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
                        decodedPoints.numInput = static_cast<uint32_t>(filteredPt.size());
                        decodedPoints.pt = std::move(filteredPt);
                        decodedPoints.att = std::move(filteredAtt);
    }

    // -----------------------------------------------------------------------------
    // Section: runOccupancyMapPipeline
    // -----------------------------------------------------------------------------

    void Pipeline::runOccupancyMapPipeline(const std::vector<int>& allowedCores) {
        setThreadAffinity(allowedCores);

        constexpr auto targetCycleDuration = std::chrono::milliseconds(100);

        while (running.load(std::memory_order_acquire)) {
            auto cycleStartTime = std::chrono::steady_clock::now();

            size_t itemsToProcess = pointsRingBufferOccMap.read_available();

            // {
            //     std::lock_guard<std::mutex> consoleLock(consoleMutex);  
            //     std::cerr << "[runOccupancyMapPipeline] itemsToProcess: " << itemsToProcess << std::endl;
            // }

            CallbackPoints::Points localPointOccMap;
            if (itemsToProcess > 0) {
                
                for (size_t i = 0; i < itemsToProcess; ++i) {
                    if (pointsRingBufferOccMap.pop(localPointOccMap)) {
                        // Keep updating localPoints with the latest current item
                    }
                }

                processPoints(localPointOccMap);

                OccupancyMapDataFrame occMapFrame;
                occMapFrame.frameID = localPointOccMap.frameID;
                occMapFrame.timestamp = localPointOccMap.t;
                occMapFrame.vehiclePosition = localPointOccMap.NED;
                occMapFrame.pointcloud.resize(localPointOccMap.numInput);

                for (uint32_t i = 0; i < localPointOccMap.numInput; ++i) {
                    occMapFrame.pointcloud[i].Pt = localPointOccMap.pt[i];
                    occMapFrame.pointcloud[i].Att = localPointOccMap.att[i];
                }
                
                occupancyMapInstance->occupancyMap(occMapFrame);

                std::vector<Voxel3D> occMapVoxels;
                occMapVoxels = occupancyMapInstance->getOccupiedVoxel();
                
                // {
                //     std::lock_guard<std::mutex> consoleLock(consoleMutex);  
                //     std::cerr << "[runOccupancyMapPipeline] occMapVoxels : " << occMapVoxels.size() << std::endl;
                // }

                if (!voxelsRingBufferOccMap.push(occMapVoxels)) {
                    if (!logQueue.push("[OccupancyMapPipeline] Voxel buffer full; data dropped!\n")) {
                        droppedLogs.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                // Create and populate report
                ReportDataFrame reportOccupancyMap;
                reportOccupancyMap.frameID = localPointOccMap.frameID;
                reportOccupancyMap.timestamp = localPointOccMap.t;
                reportOccupancyMap.numpoint = localPointOccMap.numInput;
                reportOccupancyMap.occmapsize = occMapVoxels.size();
                reportOccupancyMap.elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>( // Convert to seconds as double
                    std::chrono::steady_clock::now() - cycleStartTime).count();

                // Push report to the queue
                if (!reportOccupancyMapQueue.push(reportOccupancyMap)) {
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
                // {
                //     std::lock_guard<std::mutex> consoleLock(consoleMutex);
                //     std::cout << "[OccupancyMapPipeline] Processing Time: " 
                //             << std::chrono::duration_cast<std::chrono::milliseconds>(elapsedTime).count() << " ms\n";
                // }
                std::ostringstream oss;
                oss << "[OccupancyMapPipeline] Processing Time: " << (targetCycleDuration- elapsedTime).count() << " ms\n";
                if (!logQueue.push(oss.str())) {
                    droppedLogs.fetch_add(1, std::memory_order_relaxed);
                }
            } else if (elapsedTime > targetCycleDuration + std::chrono::milliseconds(10)) {
                // {
                //     std::lock_guard<std::mutex> consoleLock(consoleMutex);  
                //     std::cout << "Warning: [OccupancyMapPipeline] Processing took longer than 100 ms. Time: " 
                //       << std::chrono::duration_cast<std::chrono::milliseconds>(elapsedTime).count()
                //       << " ms. Skipping sleep.\n";
                // }
                std::ostringstream oss;
                oss << "Warning: [OccupancyMapPipeline] Processing exceeded target by " << (elapsedTime - targetCycleDuration).count() << " ms\n";
                if (!logQueue.push(oss.str())) {
                    droppedLogs.fetch_add(1, std::memory_order_relaxed);
                }
            }
            
        }
    }

    // -----------------------------------------------------------------------------
    // Section: runClusterExtractionPipeline
    // -----------------------------------------------------------------------------

    void Pipeline::runClusterExtractionPipeline(const std::vector<int>& allowedCores) {                       
            setThreadAffinity(allowedCores);

            constexpr auto targetCycleDuration = std::chrono::milliseconds(100);

            while (running.load(std::memory_order_acquire)) {

                auto cycleStartTime = std::chrono::steady_clock::now();

                size_t itemsToProcess = pointsRingBufferExtCls.read_available();

                // {
                //     std::lock_guard<std::mutex> consoleLock(consoleMutex);  
                //     std::cerr << "[runClusterExtractionPipeline] itemsToProcess: " << itemsToProcess << std::endl;
                // }

                CallbackPoints::Points localPointExtCls;
                if (itemsToProcess > 0) {
                    for (size_t i = 0; i < itemsToProcess; ++i) {
                        if (pointsRingBufferExtCls.pop(localPointExtCls)) {
                            // Process each frame (assuming this is intended)
                            // If you want to accumulate points, modify this logic
                        }
                    }

                    processPoints(localPointExtCls);

                    ClusterExtractorDataFrame extClsFrame;
                    extClsFrame.frameID = localPointExtCls.frameID;
                    extClsFrame.timestamp = localPointExtCls.t;
                    extClsFrame.pointcloud.resize(localPointExtCls.numInput);

                    for (uint32_t i = 0; i < localPointExtCls.numInput; ++i) {
                        extClsFrame.pointcloud[i].Pt = localPointExtCls.pt[i];
                        extClsFrame.pointcloud[i].Att = localPointExtCls.att[i];
                    }

                    // Extract clusters from the last popped frame
                    clusterExtractionInstance->extractClusters(extClsFrame);

                    std::vector<Voxel3D> extClsVoxelsPersistent;
                    std::vector<Voxel3D> extClsVoxelsNonPersistent;
                    tbb::parallel_invoke(
                        [&]() { extClsVoxelsPersistent = clusterExtractionInstance->getOccupiedVoxel(true); },
                        [&]() { extClsVoxelsNonPersistent = clusterExtractionInstance->getOccupiedVoxel(false); }
                    );
                    
                    // Push voxels to the ring buffer
                    if (!voxelsRingBufferExtClsPersistent.push(std::move(extClsVoxelsPersistent))) {
                        // if (!logQueue.push("[ClusterExtractionPipeline] Voxel buffer full; data dropped!\n")) {
                        //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
                        // }
                    }

                    if (!voxelsRingBufferExtClsNonPersistent.push(std::move(extClsVoxelsNonPersistent))) {
                        // if (!logQueue.push("[ClusterExtractionPipeline] Voxel buffer full; data dropped!\n")) {
                        //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
                        // }
                    }

                    // // Create and populate report
                    // ReportDataFrame reportClusterExtractor;
                    // reportClusterExtractor.frameID = localExtractorDataFrame.frameID;
                    // reportClusterExtractor.timestamp = localExtractorDataFrame.timestamp;
                    // reportClusterExtractor.numpoint = localExtractorDataFrame.pointcloud.size();
                    // reportClusterExtractor.occmapsize = ExtClsVoxels.size();
                    // reportClusterExtractor.elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>( // Convert to seconds as double
                    //     std::chrono::steady_clock::now() - cycleStartTime).count();

                    // // Push report to the queue
                    // if (!reportExtractClusterQueue.push(reportClusterExtractor)) {
                    //     if (!logQueue.push("[ClusterExtractionPipeline] Report queue full; data dropped!\n")) {
                    //         droppedLogs.fetch_add(1, std::memory_order_relaxed);
                    //     }
                    //     droppedExtractClusterReports.fetch_add(1, std::memory_order_relaxed); // Use specific counter
                    // }
                }

                auto cycleEndTime = std::chrono::steady_clock::now();
                auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(cycleEndTime - cycleStartTime);

                if (elapsedTime < targetCycleDuration) {
                    // std::this_thread::sleep_for(targetCycleDuration - elapsedTime);
                    // std::lock_guard<std::mutex> consoleLock(consoleMutex);
                    // std::cout << "[ClusterExtractionPipeline] Processing Time: " 
                    //         << std::chrono::duration_cast<std::chrono::milliseconds>(elapsedTime).count() << " ms\n";
                } else if (elapsedTime > targetCycleDuration + std::chrono::milliseconds(10)) {
                    {
                        // std::lock_guard<std::mutex> consoleLock(consoleMutex);  
                        // std::cout << "Warning: [ClusterExtractionPipeline] Processing took longer than 100 ms. Time: " 
                        // << std::chrono::duration_cast<std::chrono::milliseconds>(elapsedTime).count()
                        // << " ms. Skipping sleep.\n";
                    }
                    // std::ostringstream oss;
                    // oss << "Warning: [ClusterExtractionPipeline] Processing exceeded target by " 
                    //     << (elapsedTime - targetCycleDuration).count() << " ms\n";
                    // if (!logQueue.push(oss.str())) {
                    //     droppedLogs.fetch_add(1, std::memory_order_relaxed);
                    // }
                }
            }
        }

        // -----------------------------------------------------------------------------
        // Section: runVizualizationPipeline
        // -----------------------------------------------------------------------------
        
        void Pipeline::runVizualizationPipeline(const std::vector<int>& allowedCores) {
            setThreadAffinity(allowedCores);

            try {
                if (!vis.CreateVisualizerWindow("3D Voxel Visualization", 1280, 720)) {
                    return;
                }
                vis.GetRenderOption().background_color_ = Eigen::Vector3d(1, 1, 1); // White background

                // Initialize geometries with default data
                if (!voxel_grid_occMap_ptr) {
                    voxel_grid_occMap_ptr = std::make_shared<open3d::geometry::VoxelGrid>();
                    voxel_grid_occMap_ptr->origin_ = mapConfig_.mapOrigin;
                    voxel_grid_occMap_ptr->voxel_size_ = mapConfig_.resolution;
                    voxel_grid_occMap_ptr->voxels_.emplace(
                        Eigen::Vector3i(0, 0, 0), 
                        open3d::geometry::Voxel(Eigen::Vector3i(0, 0, 0), Eigen::Vector3d(1, 0, 0)) // Red voxel at origin
                    );
                }
                if (!voxel_grid_extCls_ptr) {
                    voxel_grid_extCls_ptr = std::make_shared<open3d::geometry::VoxelGrid>();
                    voxel_grid_extCls_ptr->origin_ = mapConfig_.mapOrigin;
                    voxel_grid_extCls_ptr->voxel_size_ = mapConfig_.resolution;
                    voxel_grid_extCls_ptr->voxels_.emplace(
                        Eigen::Vector3i(1, 1, 1), 
                        open3d::geometry::Voxel(Eigen::Vector3i(1, 1, 1), Eigen::Vector3d(0, 1, 0)) // Green voxel offset
                    );
                }
                if (!vehicle_mesh_ptr) {
                    vehicle_mesh_ptr = createVehicleMesh({0, 0, 0}, {0, 0, 0}); // Default vehicle at origin
                }

                // Add initial geometries
                vis.AddGeometry(voxel_grid_occMap_ptr);
                vis.AddGeometry(voxel_grid_extCls_ptr);
                vis.AddGeometry(vehicle_mesh_ptr);

                // Add coordinate frame for reference
                auto coord_frame = open3d::geometry::TriangleMesh::CreateCoordinateFrame(10.0);
                vis.AddGeometry(coord_frame);

                // Set initial camera parameters
                auto& view = vis.GetViewControl();
                view.SetLookat({0.0, 0.0, 0.0}); // Center on origin
                view.SetFront({0, 0, -1});
                view.SetUp({0, 1, 0});
                view.SetZoom(20.0); // Wide zoom

                // Register animation callback
                vis.RegisterAnimationCallback([&](open3d::visualization::Visualizer* vis_ptr) {
                    try {
                        return updateVisualization(vis_ptr);
                    } catch (const std::exception& e) {
                        return running.load(std::memory_order_acquire);
                    }
                });

                vis.Run();
                vis.DestroyVisualizerWindow();
                
            } catch (const std::exception& e) {
                vis.DestroyVisualizerWindow();
            }
        }

        // -----------------------------------------------------------------------------
        // updateVoxelGrid
        // -----------------------------------------------------------------------------

        // Helper function to update voxel grid safely
        void Pipeline::updateVoxelGrid(std::shared_ptr<open3d::geometry::VoxelGrid>& grid,
                                    const std::vector<Voxel3D>& voxels,
                                    const Eigen::Vector3d& origin,
                                    double resolution) {
            if (!voxels.empty()) {
                auto new_grid = createVoxelGrid(voxels, origin, resolution);
                if (new_grid && !new_grid->voxels_.empty()) {
                    grid->voxels_ = new_grid->voxels_; // Safe assignment
                    grid->origin_ = origin;
                    grid->voxel_size_ = resolution;
                }
            }
        }

        // -----------------------------------------------------------------------------
        // updateVehicleMesh
        // -----------------------------------------------------------------------------

        // Helper function to update vehicle mesh safely
        void Pipeline::updateVehicleMesh(std::shared_ptr<open3d::geometry::TriangleMesh>& mesh,
                                        const Eigen::Vector3d& NED, const Eigen::Vector3d& RPY) {
            auto new_mesh = createVehicleMesh(NED, RPY);
            if (new_mesh && !new_mesh->vertices_.empty()) {
                mesh->vertices_ = new_mesh->vertices_;
                mesh->triangles_ = new_mesh->triangles_;
                mesh->vertex_colors_ = new_mesh->vertex_colors_;
            }
        }

        // -----------------------------------------------------------------------------
        // updateVisualization
        // -----------------------------------------------------------------------------

        bool Pipeline::updateVisualization(open3d::visualization::Visualizer* vis) {
            bool updated = false;
            static Eigen::Vector3d currentLookat = {0, 0, 0};
            static auto lastUpdateTime = std::chrono::steady_clock::now();
            constexpr double smoothingFactor = 0.1;
            constexpr std::chrono::milliseconds targetFrameDuration(32); // ~60 FPS

            // Process Occupancy Map Voxels
            size_t itemsToProcessVoxelOccMap = voxelsRingBufferOccMap.read_available();
            size_t itemsToProcessVoxelExtClsPersistent = voxelsRingBufferExtClsPersistent.read_available();
            // {
            //     std::lock_guard<std::mutex> consoleLock(consoleMutex);  
            //     std::cerr << "[updateVisualization] itemsToProcessVoxelOccMap: " << itemsToProcessVoxelOccMap << std::endl;
            //     std::cerr << "[updateVisualization] itemsToProcessVoxelExtClsPersistent: " << itemsToProcessVoxelExtClsPersistent << std::endl;
            // }

            std::vector<Voxel3D> localVoxelProcessOccMap;
            std::vector<Voxel3D> localVoxelProcessExtClsPersistent;
            if (itemsToProcessVoxelOccMap > 0) {
                
                while (voxelsRingBufferOccMap.pop(localVoxelProcessOccMap)) {
                    // Keep latest data
                }

                // Process persistent voxels and filter using CellKey
                if (itemsToProcessVoxelExtClsPersistent > 0) {
                    while (voxelsRingBufferExtClsPersistent.pop(localVoxelProcessExtClsPersistent)) {
                        // Keep latest data
                    }

                    if (!localVoxelProcessExtClsPersistent.empty()) {
                        std::unordered_set<slam::CellKey, slam::CellKeyHash> persistentKeys;
                        persistentKeys.reserve(localVoxelProcessExtClsPersistent.size());
                        
                        for (const auto& voxel : localVoxelProcessExtClsPersistent) {
                            persistentKeys.insert(voxel.key);
                        }

                        localVoxelProcessOccMap.erase(
                            std::remove_if(localVoxelProcessOccMap.begin(), localVoxelProcessOccMap.end(),
                                [&persistentKeys](const Voxel3D& voxel) {
                                    return persistentKeys.find(voxel.key) != persistentKeys.end();
                                }),
                            localVoxelProcessOccMap.end()
                        );
                    }
                }

                if (!localVoxelProcessOccMap.empty()) {
                    updateVoxelGrid(voxel_grid_occMap_ptr, localVoxelProcessOccMap, mapConfig_.mapOrigin, mapConfig_.resolution);
                    if (vis->UpdateGeometry(voxel_grid_occMap_ptr)) {
                        updated = true;
                    }
                }
            }

            // Process Cluster Extraction Voxels
            
            size_t itemsToProcessVoxelExtClsNonPersistent = voxelsRingBufferExtClsNonPersistent.read_available();
            // {
            //     std::lock_guard<std::mutex> consoleLock(consoleMutex);  
            //     std::cerr << "[updateVisualization] itemsToProcessVoxelExtClsNonPersistent: " << itemsToProcessVoxelExtClsNonPersistent << std::endl;
            // }
            std::vector<Voxel3D> localVoxelProcessExtClsNonPersistent;
            if (itemsToProcessVoxelExtClsNonPersistent > 0) {

                while (voxelsRingBufferExtClsNonPersistent.pop(localVoxelProcessExtClsNonPersistent)) {
                    // Keep latest data
                }
                if (!localVoxelProcessExtClsNonPersistent.empty()) {
                    updateVoxelGrid(voxel_grid_extCls_ptr, localVoxelProcessExtClsNonPersistent, mapConfig_.mapOrigin, mapConfig_.resolution);
                    if (vis->UpdateGeometry(voxel_grid_extCls_ptr)) {
                        updated = true;
                    }
                }
            }

            // Process Vehicle Pose
            size_t itemsToProcessVehPose = ringBufferPose.read_available();
            // {
            //     std::lock_guard<std::mutex> consoleLock(consoleMutex);  
            //     std::cerr << "[updateVisualization] itemsToProcessVehPose: " << itemsToProcessVehPose << std::endl;
            // }
            CallbackPoints::Points localPointsVehPose;
            if (itemsToProcessVehPose > 0) {
                
                while (ringBufferPose.pop(localPointsVehPose)) {
                    // Keep latest pose
                }
                updateVehicleMesh(vehicle_mesh_ptr, localPointsVehPose.NED, localPointsVehPose.RPY);
                if (vis->UpdateGeometry(vehicle_mesh_ptr)) {
                    updated = true;
                    Eigen::Vector3d targetLookat = localPointsVehPose.NED;
                    currentLookat = currentLookat + smoothingFactor * (targetLookat - currentLookat);
                    auto& view = vis->GetViewControl();
                    view.SetLookat(currentLookat);
                    view.SetFront({0, 0, -1});
                    view.SetUp({0, 1, 0});
                    // {
                    //     std::lock_guard<std::mutex> consoleLock(consoleMutex);
                    //     std::cerr << "[updateVehicleMesh] targetLookat: " << targetLookat.transpose() << std::endl;
                    // }
                }
            }

            // Frame rate limiter
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdateTime);
            if (elapsed < targetFrameDuration) {
                std::this_thread::sleep_for(targetFrameDuration - elapsed);
            }
            lastUpdateTime = now;

            return running.load(std::memory_order_acquire) || updated;
        }

}  // namespace slam