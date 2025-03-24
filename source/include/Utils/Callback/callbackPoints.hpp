#pragma once

#include <vector>
#include <cstdint>
#include <Eigen/Dense>
#include <iostream>

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
                if (data.empty()) return;
    
                // Check the header byte to ensure data packet integrity
                if (data[0] == 0x53) { // Byte 0: Header byte
                    
                    // Extract double Timestamp (8 bytes)
                    double temp_t;
                    std::memcpy(&temp_t, &data[1], sizeof(double)); // Bytes 1 to 8

                    // Extract uint8 MaxSegment (4 byte)
                    uint32_t temp_maxSegm; 
                    std::memcpy(&temp_maxSegm, &data[9], sizeof(uint32_t)); // Bytes 9 to 12

                    // Extract uint8 SegmentIndex (4 byte)
                    uint32_t temp_segm; 
                    std::memcpy(&temp_segm, &data[13], sizeof(uint32_t)); // Bytes 13 to 16

                    // Extract position and orientation (North, East, Down, Roll, Pitch, Yaw)
                    double temp_ned[3];
                    std::memcpy(temp_ned, &data[17], 3 * sizeof(double)); // Bytes 17 to 40
                    double temp_rpy[3];
                    std::memcpy(temp_rpy, &data[41], 3 * sizeof(double)); // Bytes 41 to 64

                    // Extract frame metadata
                    uint32_t temp_frameID;
                    std::memcpy(&temp_frameID, &data[65], sizeof(uint32_t)); // Bytes 65 to 68

                    uint32_t temp_numInput;
                    std::memcpy(&temp_numInput, &data[69], sizeof(uint32_t)); // Bytes 69 to 72

                    std::cout << "[temp_frameID]: " << temp_frameID << "\n";
                    std::cout << "[frameID_]: " << frameID_ << "\n";
                    if (temp_frameID != frameID_) {
                        std::cout << "[maxNumSegment_]: " << maxNumSegment_ << "\n";
                        std::cout << "[currSegmIdx_ - 1]: " << currSegmIdx_ - 1 << "\n";
                        if (maxNumSegment_ == currSegmIdx_ - 1) {
                            std::copy(receivedPt_.begin(), receivedPt_.begin() + receivedNumInput_, points.pt.begin());
                            std::copy(receivedAtt_.begin(), receivedAtt_.begin() + receivedNumInput_, points.att.begin());

                            points.numInput = receivedNumInput_;
                            points.frameID = frameID_;
                            points.t = t_;
                            points.NED = NED_;
                            points.RPY = RPY_;
                            std::cout << "[points.numInput]: " << points.numInput << "\n";
                            std::cout << "[points.frameID]: " << points.frameID << "\n";

                        } 
                        NED_ << temp_ned[0], temp_ned[1], temp_ned[2];
                        RPY_ << temp_rpy[0], temp_rpy[1], temp_rpy[2];
                        receivedNumInput_ = 0;
                        frameID_ = temp_frameID;
                        t_ = temp_t;
                        maxNumSegment_ = temp_maxSegm;
                        currSegmIdx_ = 0;
                    }

                    if (data.size() - 73 == temp_numInput * 24) {

                        currSegmIdx_++;

                        const uint32_t temp_offset = temp_segm * 55;
                        
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
                }

            }

        private:
            double t_ ;
            std::vector<Eigen::Vector3d> receivedPt_;
            std::vector<Eigen::Vector3d> receivedAtt_;
            uint32_t receivedNumInput_ ;
            uint32_t maxNumSegment_ ;
            uint32_t currSegmIdx_ ;
            uint32_t frameID_ ;
            Eigen::Vector3d NED_ ;
            Eigen::Vector3d RPY_ ;
    };

} // namespace slam