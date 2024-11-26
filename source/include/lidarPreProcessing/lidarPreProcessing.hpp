#pragma once

#include <cstdint>
#include <deque>
#include <algorithm>
#include <cstddef>
#include <Eigen/Dense>

class lidarPreProcessing{
    public:

    private:
    std::vector<Eigen::Vector3f> points_;
    std::vector<Eigen::Vector3f> attributes_;

    void downSampling();
    void calculateSmoothness();
    void featureExtraction();
    
};