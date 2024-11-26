#include "lidarPreProcessing.hpp"

void lidarPreProcessing::calculateSmoothness(
    const std::vector<Eigen::Vector3f>& pointCloud, 
    const std::vector<float>& pointRange, 
    const std::vector<Eigen::Vector3f>& attributes, 
    std::vector<float>& cloudCurvature, 
    std::vector<int>& cloudNeighborPicked, 
    std::vector<int>& cloudLabel, 
    std::vector<std::pair<float, int>>& cloudSmoothness)
{
    int cloudSize = pointCloud.size();
    const int windowSize = 5; // Sliding window size

    for (int i = windowSize; i < cloudSize - windowSize; ++i)
    {
        // Dynamic Neighbor Weighting
        float weightSum = 0.0;
        float diffWeightedSum = 0.0;
        for (int j = -windowSize; j <= windowSize; ++j)
        {
            if (j == 0) continue; // Skip the center point
            float weight = 1.0 / (1.0 + std::abs(j)); // Weight inversely proportional to distance
            float diffRange = pointRange[i] - pointRange[i + j];
            diffWeightedSum += weight * diffRange;
            weightSum += weight;
        }
        float diffRange = diffWeightedSum / weightSum;

        // Avoid Outliers using Attribute Thresholds
        if (attributes[i].z < 0.1 || attributes[i].z > 1.0) {
            cloudCurvature[i] = FLT_MAX; // Ignore points with low or extreme reflectivity (example)
            continue;
        }

        // Normalized Curvature Calculation
        float normalizedDiff = diffRange / pointRange[i];
        cloudCurvature[i] = normalizedDiff * normalizedDiff;

        // Reset neighbor and labels
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;

        // Store smoothness for sorting
        cloudSmoothness[i].first = cloudCurvature[i];
        cloudSmoothness[i].second = i;
    }
}
