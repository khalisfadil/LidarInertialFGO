#include "DymCluster/ClusterExtraction.hpp"
#include <numeric>
#include <limits>

namespace slam {
    namespace cluster {

        // -----------------------------------------------------------------------------
        // ClusterExtraction
        // -----------------------------------------------------------------------------

        ClusterExtraction::ClusterExtraction(double resolution, 
                                            Eigen::Vector3d mapOrigin, 
                                            double tolerance,
                                            size_t min_size, 
                                            size_t max_size,
                                            size_t max_frames,
                                            unsigned int maxPointsPerVoxel,
                                            ColorMode colorMode)
            : resolution_(resolution),
            mapOrigin_(mapOrigin),
            cluster_tolerance_(tolerance),
            min_cluster_size_(min_size),
            max_cluster_size_(max_size),
            max_frames_(max_frames),
            maxPointsPerVoxel_(maxPointsPerVoxel),
            colorMode_(colorMode) {}

        // -----------------------------------------------------------------------------
        // cauchyCost
        // -----------------------------------------------------------------------------

        // Simplified Cauchy loss function
        double ClusterExtraction::cauchyCost(double error_norm, double k) {
            constexpr double eps = 1e-10;  // Prevent log(1+0) issues
            double k_inv = 1.0 / k;
            double e_div_k = error_norm * k_inv;
            return 0.5 * k * k * std::log1p(e_div_k * e_div_k + eps);
        }

        // -----------------------------------------------------------------------------
        // optimizeCentroid
        // -----------------------------------------------------------------------------

        void ClusterExtraction::optimizeCentroid(const std::map<double, Eigen::Vector3d>& prev_states,
                                Eigen::Vector3d& centroid,
                                const Eigen::Matrix3d& L,
                                double timestamp,
                                double& final_cost,
                                int max_iterations) {

            // Constants for Levenberg-Marquardt optimization
            constexpr double k = 2.0;           // Cauchy loss scale parameter
            constexpr double lambda_init = 1e-7; // Initial damping factor
            constexpr double shrink_coeff = 0.1; // Shrink factor for trust region
            constexpr double grow_coeff = 10.0; // Grow factor for trust region
            constexpr double ratio_threshold = 0.25; // Trust region update threshold
            constexpr double min_dt = 1e-6;     // Minimum time difference to avoid division by zero
            constexpr double eps = 1e-10;       // Numerical stability for Cauchy cost

            // Initial estimate and damping
            Eigen::Vector3d x = centroid;  // Current centroid estimate
            double lambda = lambda_init;   // Damping factor
            double cost = 0.0;             // Total cost (to be normalized)

            // Number of previous states for normalization
            size_t num_states = prev_states.size();
            if (num_states == 0) {  // Edge case: no previous states
                final_cost = 0.0;
                return;
            }

            // Optimization loop
            for (int iter = 0; iter < max_iterations; ++iter) {
                // Reset accumulators
                cost = 0.0;
                Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
                Eigen::Matrix3d hessian = Eigen::Matrix3d::Zero();

                // Compute residuals, cost, gradient, and Hessian
                for (const auto& [prev_time, prev_centroid] : prev_states) {
                    double dt = std::max(timestamp - prev_time, min_dt);  // Ensure positive dt
                    Eigen::Vector3d velocity = (x - prev_centroid) / dt;  // Estimated velocity
                    Eigen::Vector3d predicted = prev_centroid + velocity * dt;  // Predicted position
                    Eigen::Vector3d error = x - predicted;  // Residual
                    Eigen::Vector3d whitened_error = L * error;  // Whitened error (noise-adjusted)
                    double whitened_norm = whitened_error.norm();  // Error norm

                    // Cauchy cost contribution (sum before normalization)
                    double k_inv = 1.0 / k;
                    double e_div_k = whitened_norm * k_inv;
                    cost += 0.5 * k * k * std::log1p(e_div_k * e_div_k + eps);

                    // Compute weight for robust gradient and Hessian
                    double w = 1.0 / (1.0 + std::pow(whitened_norm / k, 2));
                    Eigen::Matrix3d J = Eigen::Matrix3d::Identity();  // Jacobian (d_error/d_x)
                    gradient += w * J.transpose() * L.transpose() * whitened_error;
                    hessian += w * J.transpose() * L.transpose() * L * J;
                }

                // Normalize cost, gradient, and Hessian by number of states
                cost /= static_cast<double>(num_states);  // Average cost per state
                gradient /= static_cast<double>(num_states);  // Normalized gradient
                hessian /= static_cast<double>(num_states);  // Normalized Hessian

                // Add damping and solve for step
                Eigen::Matrix3d H = hessian + lambda * Eigen::Matrix3d::Identity();
                Eigen::Vector3d dx = H.llt().solve(-gradient);
                if (!dx.allFinite()) {  // Check for solver failure
                    break;
                }

                // Trial step
                Eigen::Vector3d x_new = x + dx;
                double new_cost = 0.0;

                // Compute new cost for trial step
                for (const auto& [prev_time, prev_centroid] : prev_states) {
                    double dt = std::max(timestamp - prev_time, min_dt);
                    Eigen::Vector3d velocity = (x_new - prev_centroid) / dt;
                    Eigen::Vector3d predicted = prev_centroid + velocity * dt;
                    Eigen::Vector3d error = x_new - predicted;
                    double whitened_norm = (L * error).norm();
                    double k_inv = 1.0 / k;
                    double e_div_k = whitened_norm * k_inv;
                    new_cost += 0.5 * k * k * std::log1p(e_div_k * e_div_k + eps);
                }
                new_cost /= static_cast<double>(num_states);  // Normalize new cost

                // Trust region update
                double predicted_reduction = -gradient.dot(dx) - 0.5 * dx.transpose() * hessian * dx;
                double actual_reduction = cost - new_cost;
                double ratio = (predicted_reduction > 0 && actual_reduction > 0) ? actual_reduction / predicted_reduction : 0.0;

                if (ratio > ratio_threshold && new_cost < cost) {
                    x = x_new;  // Accept step
                    cost = new_cost;
                    lambda *= shrink_coeff;  // Shrink trust region
                } else {
                    lambda *= grow_coeff;  // Grow trust region
                    if (lambda > 1e10) {  // Prevent infinite growth
                        break;
                    }
                }

                // Convergence check
                if (dx.norm() < 1e-6) {
                    break;
                }
            }

            // Update output
            centroid = x;  // Final optimized centroid
            final_cost = cost;  // Final normalized cost
        }

        // -----------------------------------------------------------------------------
        // extractClusters
        // -----------------------------------------------------------------------------

        void ClusterExtraction::extractClusters(const ClusterExtractorDataFrame& frame) {
            // Early exit if input is invalid
            if (frame.pointcloud.empty()) {
                clusters_.clear();
                return;
            }

            // Alias for convenience and reuse
            const auto& points = frame.pointcloud;
            clusters_.clear();

            // Fast path for first frame
            if (prevClusters_.empty() && frame.frameID == 0) {
                extractBaseClusters(points, frame.frameID, frame.timestamp);
                if (!clusters_.empty()) {
                    prevClusters_.push_back(std::vector<slam::Cluster3D>(clusters_.begin(), clusters_.end()));  // Convert to std::vector
                    if (prevClusters_.size() > max_frames_) prevClusters_.pop_front();
                }
                return;
            }

            // Extract base clusters
            extractBaseClusters(points, frame.frameID, frame.timestamp);
            if (clusters_.empty()) return;  // Early exit if no clusters found

            // Pre-compute sizes
            const size_t curr_size = clusters_.size();
            {
                std::lock_guard<std::mutex> consoleLock(consoleMutex);  
                std::cerr << "[extractClusters] curr_size: " << curr_size << std::endl;
            }
            const size_t total_prev_size = std::accumulate(prevClusters_.begin(), prevClusters_.end(), size_t{0},
                [](size_t sum, const auto& frame_clusters) { return sum + frame_clusters.size(); });

            // Fast path for no previous clusters
            if (total_prev_size == 0) {
                for (auto& cluster : clusters_) {
                    cluster.clusterID = NewStateID();
                }
                prevClusters_.push_back(std::vector<slam::Cluster3D>(clusters_.begin(), clusters_.end()));  // Convert to std::vector
                if (prevClusters_.size() > max_frames_) prevClusters_.pop_front();
                return;
            }

            // // Prepare cost matrix and mapping
            // const size_t max_size = std::max(curr_size, total_prev_size);
            // std::vector<std::vector<double>> cost_matrix(max_size, std::vector<double>(max_size, std::numeric_limits<double>::max()));
            // std::vector<std::pair<size_t, size_t>> prev_cluster_mapping;
            // prev_cluster_mapping.reserve(total_prev_size);

            // for (size_t f = 0; f < prevClusters_.size(); ++f) {
            //     const auto& frame_clusters = prevClusters_[f];
            //     for (size_t c = 0; c < frame_clusters.size(); ++c) {
            //         prev_cluster_mapping.emplace_back(f, c);
            //     }
            // }

            // // Compute cost matrix in parallel
            // constexpr double frame_age_penalty_factor = 0.1;
            // tbb::parallel_for(tbb::blocked_range<size_t>(0, curr_size),
            //     [&](const tbb::blocked_range<size_t>& range) {
            //         for (size_t i = range.begin(); i != range.end(); ++i) {
            //             const auto& curr_cluster = clusters_[i];
            //             const Eigen::Vector3d& curr_centroid = curr_cluster.centroid;
            //             for (size_t j = 0; j < total_prev_size; ++j) {
            //                 auto [frame_idx, cluster_idx] = prev_cluster_mapping[j];
            //                 const auto& prev_cluster = prevClusters_[frame_idx][cluster_idx];
            //                 double distance = (curr_centroid - prev_cluster.centroid).norm();
            //                 if (distance <= max_distance_threshold_) {
            //                     double base_cost = calculateCost(curr_cluster, prev_cluster);
            //                     double frame_age_penalty = frame_age_penalty_factor * (prevClusters_.size() - 1 - frame_idx);
            //                     cost_matrix[i][j] = base_cost + frame_age_penalty;
            //                 }
            //             }
            //         }
            //     });

            // // Cluster association
            // std::vector<unsigned int> assignment;
            // slam::Hungarian::solve(cost_matrix, assignment);

            // std::vector<bool> matched_prev(total_prev_size, false);
            // std::vector<bool> matched_curr(curr_size, false);

            // // Assign matched IDs
            // const size_t assign_size = std::min(curr_size, assignment.size());
            // for (size_t i = 0; i < assign_size; ++i) {
            //     if (assignment[i] < total_prev_size) {
            //         size_t prev_idx = assignment[i];
            //         if (!matched_prev[prev_idx] && !matched_curr[i]) {
            //             matched_prev[prev_idx] = true;
            //             matched_curr[i] = true;
            //             auto [frame_idx, cluster_idx] = prev_cluster_mapping[prev_idx];
            //             clusters_[i].clusterID = prevClusters_[frame_idx][cluster_idx].clusterID;
            //         }
            //     }
            // }

            // // Assign new IDs to unmatched clusters
            // for (size_t i = 0; i < curr_size; ++i) {
            //     if (!matched_curr[i]) {
            //         clusters_[i].clusterID = NewStateID();
            //     }
            // }

            // // Optimize centroids and classify clusters
            // constexpr double base_static_variance = 0.01;
            // constexpr double base_dynamic_variance = 0.1;
            // constexpr double cost_threshold = 5.0;
            // constexpr double min_dt = 1e-6;

            // tbb::parallel_for(tbb::blocked_range<size_t>(0, curr_size),
            //     [&](const tbb::blocked_range<size_t>& range) {
            //         for (size_t i = range.begin(); i != range.end(); ++i) {
            //             auto& curr_cluster = clusters_[i];
            //             const unsigned int id = curr_cluster.clusterID;

            //             std::map<double, Eigen::Vector3d> prev_states;
            //             for (const auto& prev_frame : prevClusters_) {
            //                 for (const auto& prev_cluster : prev_frame) {
            //                     if (prev_cluster.clusterID == id) {
            //                         prev_states[prev_cluster.timestamp] = prev_cluster.centroid;
            //                     }
            //                 }
            //             }

            //             if (!prev_states.empty()) {
            //                 auto last_state = prev_states.rbegin();
            //                 double dt = std::max(curr_cluster.timestamp - last_state->first, min_dt);
            //                 curr_cluster.velocity = (curr_cluster.centroid - last_state->second) / dt;

            //                 if (prev_states.size() > 1) {
            //                     const double size_factor = 1.0 / std::sqrt(static_cast<double>(curr_cluster.idx.size()));
            //                     const Eigen::Matrix3d static_noise = Eigen::Matrix3d::Identity() * (base_static_variance * size_factor);
            //                     const Eigen::Matrix3d dynamic_noise = Eigen::Matrix3d::Identity() * (base_dynamic_variance * size_factor);

            //                     const Eigen::LLT<Eigen::Matrix3d> static_llt(static_noise.inverse());
            //                     const Eigen::LLT<Eigen::Matrix3d> dynamic_llt(dynamic_noise.inverse());
            //                     const Eigen::Matrix3d static_L = static_llt.matrixL().transpose();
            //                     const Eigen::Matrix3d dynamic_L = dynamic_llt.matrixL().transpose();

            //                     Eigen::Vector3d static_centroid = curr_cluster.centroid;
            //                     Eigen::Vector3d dynamic_centroid = curr_cluster.centroid;
            //                     double static_cost = 0.0, dynamic_cost = 0.0;

            //                     tbb::parallel_invoke(
            //                         [&]() { optimizeCentroid(prev_states, static_centroid, static_L, curr_cluster.timestamp, static_cost); },
            //                         [&]() { optimizeCentroid(prev_states, dynamic_centroid, dynamic_L, curr_cluster.timestamp, dynamic_cost); }
            //                     );

            //                     if (static_cost < cost_threshold && dynamic_cost < cost_threshold) {
            //                         curr_cluster.isDynamic = (dynamic_cost < static_cost);
            //                         curr_cluster.centroid = (dynamic_cost < static_cost) ? dynamic_centroid : static_centroid;
            //                     } else {
            //                         curr_cluster.isDynamic = false;
            //                         curr_cluster.centroid = static_centroid;
            //                     }
            //                 }
            //             } else {
            //                 curr_cluster.isDynamic = false;
            //             }
            //         }
            //     });

            // // Extract dynamic points with pre-check
            // dynamic_points_.clear();
            // tbb::concurrent_vector<slam::Point3D> concurrent_dynamic_points;
            // std::atomic<bool> has_dynamic{false};  // Use atomic for thread safety

            // tbb::parallel_for(tbb::blocked_range<size_t>(0, curr_size),
            //     [&](const tbb::blocked_range<size_t>& range) {
            //         for (size_t i = range.begin(); i != range.end(); ++i) {
            //             const auto& cluster = clusters_[i];
            //             if (cluster.isDynamic) {
            //                 has_dynamic.store(true, std::memory_order_relaxed);  // Thread-safe update
            //                 for (size_t idx : cluster.idx) {
            //                     concurrent_dynamic_points.push_back(points[idx]);
            //                 }
            //             }
            //         }
            //     });

            // // Only proceed with assignment and occupancy map if we have dynamic points
            // if (has_dynamic.load(std::memory_order_relaxed)) {  // Thread-safe read
            //     dynamic_points_.assign(concurrent_dynamic_points.begin(), concurrent_dynamic_points.end());
                
            //     // Parallel invocation to update both persistent and non-persistent maps concurrently
            //     tbb::parallel_invoke(
            //         [&]() { clusterOccupancyMapBase(dynamic_points_, frame.frameID, frame.timestamp, true); },  // Persistent map (history)
            //         [&]() { clusterOccupancyMapBase(dynamic_points_, frame.frameID, frame.timestamp, false); }  // Non-persistent map (latest)
            //     );
            // }

            // Update sliding window
            if (!clusters_.empty()) {
                prevClusters_.push_back(std::vector<slam::Cluster3D>(clusters_.begin(), clusters_.end()));  // Convert to std::vector
                if (prevClusters_.size() > max_frames_) prevClusters_.pop_front();
            }
        }
        // -----------------------------------------------------------------------------
        // extractBaseClusters
        // -----------------------------------------------------------------------------

        void ClusterExtraction::extractBaseClusters(const std::vector<Point3D>& points, unsigned int frame_id, double timestamp) {
            if (points.empty()) return;

            Eigen::Vector3d ori = Eigen::Vector3d::Zero();

            using GridType = tbb::concurrent_unordered_map<CellKey, tbb::concurrent_vector<size_t>, CellKeyHash>;
            GridType grid;
            std::vector<CellKey> point_to_cell(points.size());
            tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        CellKey key = CellKey::fromPoint(points[i].Pt, ori, resolution_);
                        point_to_cell[i] = key;
                        grid[key].push_back(i);
                    }
                });

            tbb::concurrent_vector<std::pair<size_t, size_t>> edges;
            edges.reserve(points.size());
            tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        const CellKey& cell = point_to_cell[i];
                        for (int dx = -1; dx <= 1; ++dx) {
                            for (int dy = -1; dy <= 1; ++dy) {
                                for (int dz = -1; dz <= 1; ++dz) {
                                    CellKey neighbor{cell.x + dx, cell.y + dy, cell.z + dz};
                                    if (auto it = grid.find(neighbor); it != grid.end()) {
                                        for (size_t j : it->second) {
                                            if (i < j && (points[i].Pt - points[j].Pt).norm() <= cluster_tolerance_) {
                                                edges.emplace_back(i, j);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                });

            UnionFind uf(points.size());
            for (const auto& edge : edges) {
                uf.unite(edge.first, edge.second);
            }

            using ClusterMap = tbb::concurrent_unordered_map<size_t, tbb::concurrent_vector<size_t>>;
            ClusterMap cluster_map;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        cluster_map[uf.find(i)].push_back(i);
                    }
                });

            clusters_.clear();
            clusters_.reserve(cluster_map.size());  // Optional pre-allocation

            tbb::parallel_for_each(cluster_map.begin(), cluster_map.end(),
                [&](const auto& pair) {
                    const auto& [root, indices] = pair;
                    if (indices.size() >= min_cluster_size_ && indices.size() <= max_cluster_size_ && !indices.empty()) {
                        slam::Cluster3D cluster;
                        cluster.idx = std::vector<size_t>(indices.begin(), indices.end());
                        cluster.clusterID = std::numeric_limits<unsigned int>::max();
                        cluster.frameID = frame_id;
                        cluster.timestamp = timestamp;

                        cluster.centroid.setZero();
                        cluster.averageAtt.setZero();
                        bool valid = true;
                        for (size_t idx : cluster.idx) {
                            if (idx >= points.size()) {
                                valid = false;
                                break;
                            }
                            cluster.centroid += points[idx].Pt;
                            cluster.averageAtt += points[idx].Att;
                        }
                        if (!valid) return;

                        double cluster_size = static_cast<double>(cluster.idx.size());
                        cluster.centroid /= cluster_size;
                        cluster.averageAtt /= cluster_size;

                        if (cluster.idx[0] >= points.size()) return;
                        cluster.minBound = points[cluster.idx[0]].Pt;
                        cluster.maxBound = points[cluster.idx[0]].Pt;

                        for (size_t idx : cluster.idx) {
                            if (idx >= points.size()) {
                                valid = false;
                                break;
                            }
                            cluster.minBound = cluster.minBound.cwiseMin(points[idx].Pt);
                            cluster.maxBound = cluster.maxBound.cwiseMax(points[idx].Pt);
                        }
                        if (!valid) return;

                        clusters_.push_back(std::move(cluster));  // Thread-safe push_back
                    }
                });
        }

        // -----------------------------------------------------------------------------
        // calculateCost
        // -----------------------------------------------------------------------------

        double ClusterExtraction::calculateCost(const slam::Cluster3D& clusterA, const slam::Cluster3D& clusterB) const {
            constexpr double distance_weight = 0.3;
            constexpr double bbox_weight = 0.3;
            constexpr double att_weight = 0.2;
            constexpr double velocity_weight = 0.2;
            constexpr double min_dt = 1e-6;

            double distance = (clusterA.centroid - clusterB.centroid).norm();
            Eigen::Vector3d intersection_min = clusterA.minBound.cwiseMax(clusterB.minBound);
            Eigen::Vector3d intersection_max = clusterA.maxBound.cwiseMin(clusterB.maxBound);
            Eigen::Vector3d intersection_size = (intersection_max - intersection_min).cwiseMax(Eigen::Vector3d::Zero());
            double intersection_vol = intersection_size.prod();
            double volA = (clusterA.maxBound - clusterA.minBound).prod();
            double volB = (clusterB.maxBound - clusterB.minBound).prod();
            double union_vol = volA + volB - intersection_vol;
            double iou = union_vol > 0 ? intersection_vol / union_vol : 0.0;
            double att_diff = (clusterA.averageAtt - clusterB.averageAtt).norm();

            double dt = std::max(clusterA.timestamp - clusterB.timestamp, min_dt);
            Eigen::Vector3d v_estimated = (clusterA.centroid - clusterB.centroid) / dt;
            Eigen::Vector3d v_predicted = clusterA.velocity;  // Use clusterA's velocity as prediction
            double v_diff = (v_estimated - v_predicted).norm();

            return distance_weight * distance + bbox_weight * (1.0 - iou) + 
                att_weight * att_diff + velocity_weight * v_diff;
        }

        // -----------------------------------------------------------------------------
        // Section: assignVoxelColorsRed
        // -----------------------------------------------------------------------------

        void ClusterExtraction::clusterOccupancyMapBase(const std::vector<Point3D>& points, unsigned int frame_id, double timestamp, bool tracked) {
            if (points.empty()) return;

            occupancyMap_.clear();

            // Choose the working map: persistentMap_ for tracked (history), tempMap for non-tracked (one-time)
            GridType& workingMap = tracked ? persistentMap_ : occupancyMap_;

            std::vector<CellKey> point_to_cell(points.size());

            tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        CellKey key = CellKey::fromPoint(points[i].Pt, mapOrigin_, resolution_);
                        point_to_cell[i] = key;

                        auto result = workingMap.insert({key, Voxel3D{}});
                        Voxel3D& voxel = result.first->second;

                        if (voxel.counter < maxPointsPerVoxel_) {
                            unsigned int newCount = voxel.counter + 1;
                            voxel.averagePointPose = (voxel.averagePointPose * voxel.counter + points[i].Pt) / newCount;
                            voxel.averagePointAtt = (voxel.averagePointAtt * voxel.counter + points[i].Att) / newCount;
                            voxel.counter = newCount;
                            voxel.frameID = frame_id;
                            voxel.timestamp = timestamp;
                            voxel.key = key;

                            // Switch based on color mode
                            switch (colorMode_) {
                                case ColorMode::Occupancy:
                                    voxel.color = computeOccupancyColor(newCount);
                                    break;
                                case ColorMode::Reflectivity:
                                    voxel.color = computeReflectivityColor(voxel.averagePointAtt.x());
                                    break;
                                case ColorMode::Intensity:
                                    voxel.color = computeIntensityColor(voxel.averagePointAtt.y());
                                    break;
                                case ColorMode::NIR:
                                    voxel.color = computeNIRColor(voxel.averagePointAtt.z());
                                    break;
                                default:  // Shouldn’t happen, but default to Occupancy
                                    voxel.color = computeOccupancyColor(newCount);
                                    break;
                            }
                        } else {
                            voxel.frameID = frame_id;
                            voxel.timestamp = timestamp;
                        }
                    }
                });

            // Note: tempMap is discarded when tracked is false; persistentMap_ accumulates history when tracked is true
            // Optionally update occupancyMap_ with tempMap results for non-tracked case if desired
            // if (!tracked) {
            //     occupancyMap_ = std::move(tempMap);  // Uncomment if you want non-tracked results in occupancyMap_
            // }
        }

        // -----------------------------------------------------------------------------
        // Section: assignVoxelColorsRed
        // -----------------------------------------------------------------------------

        Eigen::Vector3i ClusterExtraction::computeOccupancyColor(unsigned int counter) const {
            int value = (255 * std::min(counter, maxPointsPerVoxel_)) / maxPointsPerVoxel_;
            return Eigen::Vector3i(value, 0, 0);  // Red scale: only R varies, G and B are 0
        }

        // -----------------------------------------------------------------------------
        // Section: assignVoxelColorsRed
        // -----------------------------------------------------------------------------

        Eigen::Vector3i ClusterExtraction::computeReflectivityColor(double avgReflectivity) const {
            int reflectivityColorValue;
            if (avgReflectivity <= 100.0) {
                reflectivityColorValue = static_cast<int>(avgReflectivity * 2.55);
            } else {
                float transitionFactor = 0.2;
                if (avgReflectivity <= 110.0) {
                    float linearComponent = 2.55 * avgReflectivity;
                    float logComponent = 155.0 + (100.0 * (std::log2(avgReflectivity - 100.0 + 1.0) / std::log2(156.0)));
                    reflectivityColorValue = static_cast<int>((1.0 - transitionFactor) * linearComponent + transitionFactor * logComponent);
                } else {
                    float logReflectivity = std::log2(avgReflectivity - 100.0 + 1.0) / std::log2(156.0);
                    reflectivityColorValue = static_cast<int>(155.0 + logReflectivity * 100.0);
                }
            }
            reflectivityColorValue = std::clamp(reflectivityColorValue, 0, 255);
            return Eigen::Vector3i(reflectivityColorValue, 0, 0);  // Red scale: only R varies, G and B are 0
        }

        // -----------------------------------------------------------------------------
        // Section: calculateIntensityColor
        // -----------------------------------------------------------------------------

        Eigen::Vector3i ClusterExtraction::computeIntensityColor(double avgIntensity) const {
            int intensityColorValue = static_cast<int>(std::clamp(avgIntensity, 0.0, 255.0));
            return Eigen::Vector3i(intensityColorValue, 0, 0);  // Red scale: only R varies, G and B are 0
        }

        // -----------------------------------------------------------------------------
        // Section: calculateNIRColor
        // -----------------------------------------------------------------------------

        Eigen::Vector3i ClusterExtraction::computeNIRColor(double avgNIR) const {
            int NIRColorValue = static_cast<int>(std::clamp(avgNIR, 0.0, 255.0));
            return Eigen::Vector3i(NIRColorValue, 0, 0);  // Red scale: only R varies, G and B are 0
        }

        // -----------------------------------------------------------------------------
        // Section: calculateNIRColor
        // -----------------------------------------------------------------------------

        std::vector<Voxel3D> ClusterExtraction::getOccupiedVoxel(bool tracked) const {
            // Use a tbb::concurrent_vector for thread-safe parallel collection
            tbb::concurrent_vector<Voxel3D> occupiedVoxels;
            
            // Choose the working map based on tracked parameter
            const GridType& workingMap = tracked ? persistentMap_ : occupancyMap_;
            
            // Reserve space to reduce reallocations
            occupiedVoxels.reserve(workingMap.size());

            // Parallel iteration over the concurrent map
            tbb::parallel_for_each(workingMap.begin(), workingMap.end(),
                [&](const auto& mapEntry) {
                    const Voxel3D& voxel = mapEntry.second;
                    if (voxel.counter > 0) {
                        occupiedVoxels.push_back(voxel); // Thread-safe push_back
                    }
                });

            // Convert to std::vector and return
            return std::vector<Voxel3D>(occupiedVoxels.begin(), occupiedVoxels.end());
        }
    }  // namespace cluster
}  // namespace slam