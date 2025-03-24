#include "DymMap/Pipeline.hpp"

using namespace slam;

int main() {
    // Get the singleton instance of Pipeline
    // Pipeline& pipeline = Pipeline::getInstance();

    Pipeline pipeline;

    // Set up signal handling
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = Pipeline::signalHandler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, nullptr);
    sigaction(SIGTERM, &sigIntHandler, nullptr);

    std::cout << "[Main] Starting pipeline processes..." << std::endl;

    try {
        std::vector<std::thread> threads;

        // Listener configurations
        std::string pointsHost = "127.0.0.1";
        uint16_t pointsPort = 61234;
        uint32_t bufferSize = 1393;

        // Start points listener
        boost::asio::io_context ioContextPoints;
        threads.emplace_back(
            [&]() {
                pipeline.startPointsListener(ioContextPoints,
                                             pointsHost,
                                             pointsPort,
                                             bufferSize,
                                             std::vector<int>{12});
            }
        );

        // // Start Occupancy Map Pipeline
        // threads.emplace_back(
        //     [&]() {
        //         pipeline.runOccupancyMapPipeline(std::vector<int>{0, 1, 2, 3});
        //     }
        // );

        // // Start Cluster Extraction Pipeline
        // threads.emplace_back(
        //     [&]() {
        //         pipeline.runClusterExtractionPipeline(std::vector<int>{4, 5, 6, 7});
        //     }
        // );

        // // Start Visualization Pipeline
        // threads.emplace_back(
        //     [&]() {
        //         pipeline.runVizualizationPipeline(std::vector<int>{8, 9, 10, 11});
        //     }
        // );

        // // Start logging thread
        // threads.emplace_back(
        //     [&]() {
        //         pipeline.processLogQueue(std::vector<int>{20});
        //     }
        // );

        // // Start Occupancy Map Report Queue Processing
        // threads.emplace_back(
        //     [&]() {
        //         pipeline.processReportQueueOccMap("../source/result/occupancy_report.txt", 
        //                                           std::vector<int>{21});
        //     }
        // );

        // // Start Cluster Extraction Report Queue Processing
        // threads.emplace_back(
        //     [&]() {
        //         pipeline.processReportQueueExtCls("../source/result/cluster_report.txt", 
        //                                           std::vector<int>{22});
        //     }
        // );

        // Monitor signal and clean up
        while (Pipeline::running.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Stop IO context
        ioContextPoints.stop();

        // Join all threads
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: [Main] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[Main] All processes stopped. Exiting program." << std::endl;
    return EXIT_SUCCESS;
}