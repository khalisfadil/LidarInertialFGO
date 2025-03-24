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

                if (data.size() != 73 + temp_numInput * 24) return;

                if (temp_frameID != frameID_) {
                    if (maxNumSegment_ == currSegmIdx_ - 1 && receivedNumInput_ > 0) {
                        points.pt.assign(receivedPt_.begin(), receivedPt_.begin() + receivedNumInput_);
                        points.att.assign(receivedAtt_.begin(), receivedAtt_.begin() + receivedNumInput_);
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

                currSegmIdx_++;
                const uint32_t temp_offset = temp_segm * 57;
                if (temp_offset + temp_numInput > MAX_NUM_POINT) return;

                const float* pointData = reinterpret_cast<const float*>(&data[73]);
                if (temp_numInput >= 16) {
                    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, temp_numInput, 4),
                        [&](const tbb::blocked_range<uint32_t>& r) {
                            for (uint32_t i = r.begin(); i < r.end(); i += 4) {
                                if (i + 3 < r.end()) {
                                    // Load 24 floats (4 points × 6 floats)
                                    __m256 p1 = _mm256_loadu_ps(pointData + i * 6);     // x1, y1, z1, ax1, ay1, az1, x2, y2
                                    __m256 p2 = _mm256_loadu_ps(pointData + i * 6 + 8); // z2, ax2, ay2, az2, x3, y3, z3, ax3
                                    __m256 p3 = _mm256_loadu_ps(pointData + i * 6 + 16);// ay3, az3, x4, y4, z4, ax4, ay4, az4

                                    // Convert to double
                                    __m256d pt1 = _mm256_cvtps_pd(_mm256_extractf128_ps(p1, 0)); // x1, y1, z1, ax1
                                    __m256d pt2 = _mm256_cvtps_pd(_mm256_extractf128_ps(p2, 0)); // z2, ax2, ay2, az2
                                    __m256d pt3 = _mm256_cvtps_pd(_mm256_extractf128_ps(p3, 0)); // ay3, az3, x4, y4

                                    __m256d att1 = _mm256_cvtps_pd(_mm256_extractf128_ps(p1, 1)); // ay1, az1, x2, y2
                                    __m256d att2 = _mm256_cvtps_pd(_mm256_extractf128_ps(p2, 1)); // x3, y3, z3, ax3
                                    __m256d att3 = _mm256_cvtps_pd(_mm256_extractf128_ps(p3, 1)); // z4, ax4, ay4, az4

                                    // Store
                                    _mm256_storeu_pd(receivedPt_[temp_offset + i].data(), pt1);
                                    _mm256_storeu_pd(receivedPt_[temp_offset + i + 1].data() + 2, pt2); // Offset for z
                                    _mm256_storeu_pd(receivedPt_[temp_offset + i + 2].data() + 1, pt3); // Offset for y,z

                                    _mm256_storeu_pd(receivedAtt_[temp_offset + i].data() + 1, att1);    // Offset for y,z
                                    _mm256_storeu_pd(receivedAtt_[temp_offset + i + 1].data(), att2);
                                    _mm256_storeu_pd(receivedAtt_[temp_offset + i + 2].data() + 2, att3); // Offset for z
                                } else {
                                    for (; i < r.end(); ++i) {
                                        const float* base = pointData + i * 6;
                                        receivedPt_[temp_offset + i] = Eigen::Map<const Eigen::Vector3f, Eigen::Unaligned>(base).cast<double>();
                                        receivedAtt_[temp_offset + i] = Eigen::Map<const Eigen::Vector3f, Eigen::Unaligned>(base + 3).cast<double>();
                                    }
                                }
                            }
                        }, tbb::simple_partitioner());
                } else {
                    for (uint32_t i = 0; i < temp_numInput; ++i) {
                        const float* base = pointData + i * 6;
                        receivedPt_[temp_offset + i] = Eigen::Map<const Eigen::Vector3f, Eigen::Unaligned>(base).cast<double>();
                        receivedAtt_[temp_offset + i] = Eigen::Map<const Eigen::Vector3f, Eigen::Unaligned>(base + 3).cast<double>();
                    }
                }

                receivedNumInput_ = temp_offset + temp_numInput;
                std::cout << "[startPointsListener]: " << receivedNumInput_ << "\n";
                std::cout << "[startPointsListener]: " << points.numInput << "\n";
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