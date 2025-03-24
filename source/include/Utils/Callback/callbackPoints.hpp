#pragma once

#include <vector>
#include <cstdint>
#include <Eigen/Dense>
#include <tbb/parallel_for.h>
#include <immintrin.h>

#include "Utils/constants.hpp"

namespace slam {

    class CallbackPoints {
        public:
            // -----------------------------------------------------------------------------
            /**
             * @struct Points
             * 
             * @brief Represents a container for points data.
             * Contains 3D points, frame metadata, and associated positional/orientation data.
             */
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

            // -----------------------------------------------------------------------------
            CallbackPoints() 
                : receivedPt_(MAX_NUM_POINT, Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN())),
                receivedAtt_(MAX_NUM_POINT, Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()))
            {}

            // -----------------------------------------------------------------------------
            void process(const std::vector<uint8_t>& data, Points& points) {
                if (data.size() < 73 || data[0] != 0x53) return;

                const double temp_t = *reinterpret_cast<const double*>(&data[1]);
                const uint32_t temp_maxSegm = *reinterpret_cast<const uint32_t*>(&data[9]);
                const uint32_t temp_segm = *reinterpret_cast<const uint32_t*>(&data[13]);
                const uint32_t temp_frameID = *reinterpret_cast<const uint32_t*>(&data[65]);
                const uint32_t temp_numInput = *reinterpret_cast<const uint32_t*>(&data[69]);
                std::cout << "[startPointsListener1]: " << temp_frameID << "\n";
                if (data.size() != 73 + temp_numInput * 24) return;

                if (temp_frameID != frameID_) {
                    if (maxNumSegment_ == currSegmIdx_ - 1) {
                        std::copy(receivedPt_.begin(), receivedPt_.begin() + receivedNumInput_, points.pt.begin());
                        std::copy(receivedAtt_.begin(), receivedAtt_.begin() + receivedNumInput_, points.att.begin());

                        points.numInput = receivedNumInput_;
                        points.frameID = frameID_;
                        points.t = t_;
                        points.NED = NED_;
                        points.RPY = RPY_;
                    } 
                    NED_ = Eigen::Map<const Eigen::Vector3d, Eigen::Unaligned>(reinterpret_cast<const double*>(&data[17]));
                    RPY_ = Eigen::Map<const Eigen::Vector3d, Eigen::Unaligned>(reinterpret_cast<const double*>(&data[41]));
                    receivedNumInput_ = 0;
                    frameID_ = temp_frameID;
                    t_ = temp_t;
                    maxNumSegment_ = temp_maxSegm;
                    currSegmIdx_ = 0;
                }
                

                if (data.size() - 73 == temp_numInput * 24) {

                    currSegmIdx_++;
                    const uint32_t temp_offset = temp_segm * 57;
                    
                    for (uint32_t i = 0; i < temp_numInput; ++i) {
                        float point[3];    // For XYZ coordinates
                        float attrib[3];   // For attributes
                        
                        // Copy XYZ coordinates (first 12 bytes = 3 floats)
                        std::memcpy(point, &data[73 + (i * 12)], sizeof(point));
                        receivedPt_[temp_offset + i] = Eigen::Vector3d(point[0], point[1], point[2]);
                        
                        // Copy attributes (next 12 bytes = 3 floats)
                        std::memcpy(attrib, &data[73 + (i * 12) + 12], sizeof(attrib));
                        receivedAtt_[temp_offset + i] = Eigen::Vector3d(attrib[0], attrib[1], attrib[2]);
                    }
                    
                    receivedNumInput_ = temp_offset + temp_numInput;
                }
                std::cout << "[startPointsListener2]: " << points.frameID << "\n";
            }

        private:
            double t_ = 0.0;
            std::vector<Eigen::Vector3d> receivedPt_;
            std::vector<Eigen::Vector3d> receivedAtt_;
            uint32_t receivedNumInput_ = 0;
            uint32_t maxNumSegment_ = 0;
            uint32_t currSegmIdx_ = 0;
            uint32_t frameID_ = 0;
            Eigen::Vector3d NED_ = Eigen::Vector3d::Zero();
            Eigen::Vector3d RPY_ = Eigen::Vector3d::Zero();
    };

} // namespace slam