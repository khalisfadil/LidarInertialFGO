#include "DymMap/Pipeline.hpp"

using namespace slam;

int main() {
    Pipeline pipeline;

    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = [](int signal) { Pipeline::signalHandler(signal); };
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, nullptr);
    sigaction(SIGTERM, &sigIntHandler, nullptr);

    std::cout << "[Main] Starting pipeline processes..." << std::endl;

    try {
        std::vector<std::thread> threads;
        std::string pointsHost = "127.0.0.1";
        uint16_t pointsPort = 61234;
        uint32_t bufferSize = 1393;

        boost::asio::io_context ioContextPoints;
        threads.emplace_back([&]() {
            pipeline.startPointsListener(ioContextPoints, pointsHost, pointsPort, bufferSize, {12, 13, 14, 15});
        });
        threads.emplace_back([&]() { pipeline.runOccupancyMapPipeline({0, 1, 2, 3}); });
        threads.emplace_back([&]() { pipeline.runClusterExtractionPipeline({4, 5, 6, 7}); });
        threads.emplace_back([&]() { pipeline.runVizualizationPipeline({8, 9, 10, 11}); });
        threads.emplace_back([&]() { pipeline.processLogQueue({20}); });
        threads.emplace_back([&]() { pipeline.processReportQueueOccMap("../source/result/occupancy_report.txt", {21}); });
        threads.emplace_back([&]() { pipeline.processReportQueueExtCls("../source/result/cluster_report.txt", {22}); });

        while (pipeline.running_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        ioContextPoints.stop();
        for (auto& thread : threads) {
            if (thread.joinable()) thread.join();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: [Main] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[Main] All processes stopped. Exiting program." << std::endl;
    return EXIT_SUCCESS;
}