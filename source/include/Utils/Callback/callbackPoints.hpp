#pragma once

#include <vector>
#include <cstdint>
#include <Eigen/Dense>
#include <cstring>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "Utils/constants.hpp"

namespace slam {
    class CallbackPoints {
    public:
        struct Points {
            std::vector<Eigen::Vector3d> pt;
            std::vector<Eigen::Vector3d> att;
            uint32_t numInput = 0;
            uint32_t frameID = 0;
            double t = 0.0;
            Eigen::Vector3d NED = Eigen::Vector3d::Zero();
            Eigen::Vector3d RPY = Eigen::Vector3d::Zero();

            Points() : pt(MAX_NUM_POINT, Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN())),
                      att(MAX_NUM_POINT, Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN())) {}
        };

        CallbackPoints() 
            : receivedPt_(MAX_NUM_POINT, Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN())),
              receivedAtt_(MAX_NUM_POINT, Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN())),
              t_(0.0), receivedNumInput_(0), maxNumSegment_(0), currSegmIdx_(0), frameID_(0) {}

        void process(const std::vector<uint8_t>& data, Points& points) noexcept {
            if (data.size() < 73 || data[0] != 0x53) return;

            const uint8_t* buffer = data.data();
            const double t = *reinterpret_cast<const double*>(buffer + 1);
            const uint32_t maxSegm = *reinterpret_cast<const uint32_t*>(buffer + 9);
            const uint32_t segmIdx = *reinterpret_cast<const uint32_t*>(buffer + 13);
            const double* ned = reinterpret_cast<const double*>(buffer + 17);
            const double* rpy = reinterpret_cast<const double*>(buffer + 41);
            const uint32_t frameId = *reinterpret_cast<const uint32_t*>(buffer + 65);
            const uint32_t numInput = *reinterpret_cast<const uint32_t*>(buffer + 69);

            if (frameId != frameID_) {
                if (maxNumSegment_ == currSegmIdx_ - 1) {
                    // Serial copy for small datasets
                    std::copy_n(receivedPt_.data(), receivedNumInput_, points.pt.data());
                    std::copy_n(receivedAtt_.data(), receivedNumInput_, points.att.data());
                    points.numInput = receivedNumInput_;
                    points.frameID = frameID_;
                    points.t = t_;
                    points.NED = NED_;
                    points.RPY = RPY_;
                }

                NED_.x() = ned[0]; NED_.y() = ned[1]; NED_.z() = ned[2];
                RPY_.x() = rpy[0]; RPY_.y() = rpy[1]; RPY_.z() = rpy[2];
                
                receivedNumInput_ = 0;
                frameID_ = frameId;
                t_ = t;
                maxNumSegment_ = maxSegm;
                currSegmIdx_ = 0;
            }

            if (data.size() != 73 + numInput * 24) return;

            currSegmIdx_++;
            const uint32_t offset = segmIdx * 55;
            const float* payload = reinterpret_cast<const float*>(buffer + 73);

            // Sequential loop replacing tbb::parallel_for
            for (uint32_t i = 0; i < numInput; ++i) {
                const float* base = payload + i * 6;
                receivedPt_[offset + i] = Eigen::Vector3d(
                    static_cast<double>(base[0]),
                    static_cast<double>(base[1]),
                    static_cast<double>(base[2])
                );
                receivedAtt_[offset + i] = Eigen::Vector3d(
                    static_cast<double>(base[3]),
                    static_cast<double>(base[4]),
                    static_cast<double>(base[5])
                );
            }

            receivedNumInput_ = offset + numInput;
        }

    private:
        alignas(64) std::vector<Eigen::Vector3d> receivedPt_;
        alignas(64) std::vector<Eigen::Vector3d> receivedAtt_;
        double t_;
        uint32_t receivedNumInput_;
        uint32_t maxNumSegment_;
        uint32_t currSegmIdx_;
        uint32_t frameID_;
        Eigen::Vector3d NED_;
        Eigen::Vector3d RPY_;
    };
} // namespace slam