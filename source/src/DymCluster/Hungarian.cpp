#include "Hungarian.hpp"
#include <algorithm>
#include <numeric>
#include <limits>

namespace slam {

    double Hungarian::solve(const std::vector<std::vector<double>>& costMatrix, 
                            std::vector<unsigned int>& assignment) {
        if (costMatrix.empty() || costMatrix[0].empty()) {
            assignment.clear();
            return 0.0;
        }

        size_t rows = costMatrix.size();
        size_t cols = costMatrix[0].size();
        std::vector<std::vector<double>> cost = costMatrix;

        // Step 1: Subtract row minima in parallel
        subtractRowMinima(cost);

        // Step 2: Subtract column minima in parallel
        subtractColMinima(cost);

        // Step 3: Assign as many zeros as possible
        std::vector<unsigned int> rowAssign(rows, std::numeric_limits<unsigned int>::max());
        std::vector<unsigned int> colAssign(cols, std::numeric_limits<unsigned int>::max());
        std::vector<bool> rowCovered(rows, false);
        std::vector<bool> colCovered(cols, false);

        while (!findAssignment(cost, rowAssign, colAssign, rowCovered, colCovered)) {
            coverColumns(cost, rowAssign, colAssign, rowCovered, colCovered);
            adjustUncovered(cost, rowCovered, colCovered);
            std::fill(rowCovered.begin(), rowCovered.end(), false);
            std::fill(colCovered.begin(), colCovered.end(), false);
        }

        // Compute total cost and prepare assignment
        double totalCost = 0.0;
        assignment.resize(rows, std::numeric_limits<unsigned int>::max());
        for (size_t i = 0; i < rows; ++i) {
            if (rowAssign[i] != std::numeric_limits<unsigned int>::max()) {
                assignment[i] = rowAssign[i];
                totalCost += costMatrix[i][rowAssign[i]];
            }
        }

        return totalCost;
    }

    void Hungarian::subtractRowMinima(std::vector<std::vector<double>>& cost) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, cost.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    double minVal = *std::min_element(cost[i].begin(), cost[i].end());
                    if (minVal != 0) {
                        for (double& val : cost[i]) {
                            val -= minVal;
                        }
                    }
                }
            });
    }

    void Hungarian::subtractColMinima(std::vector<std::vector<double>>& cost) {
        size_t cols = cost[0].size();
        for (size_t j = 0; j < cols; ++j) {
            double minVal = std::numeric_limits<double>::max();
            for (size_t i = 0; i < cost.size(); ++i) {
                minVal = std::min(minVal, cost[i][j]);
            }
            if (minVal != 0) {
                for (size_t i = 0; i < cost.size(); ++i) {
                    cost[i][j] -= minVal;
                }
            }
        }
    }

    bool Hungarian::findAssignment(std::vector<std::vector<double>>& cost, 
                                   std::vector<unsigned int>& rowAssign, 
                                   std::vector<unsigned int>& colAssign, 
                                   std::vector<bool>& rowCovered, 
                                   std::vector<bool>& colCovered) {
        std::fill(rowAssign.begin(), rowAssign.end(), std::numeric_limits<unsigned int>::max());
        std::fill(colAssign.begin(), colAssign.end(), std::numeric_limits<unsigned int>::max());

        tbb::spin_mutex mutex;
        bool done = false;
        while (!done) {
            done = true;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, cost.size()),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        if (rowAssign[i] == std::numeric_limits<unsigned int>::max()) {
                            for (size_t j = 0; j < cost[i].size(); ++j) {
                                if (cost[i][j] == 0 && colAssign[j] == std::numeric_limits<unsigned int>::max() && !colCovered[j]) {
                                    tbb::spin_mutex::scoped_lock lock(mutex);
                                    if (colAssign[j] == std::numeric_limits<unsigned int>::max()) { // Double-check under lock
                                        rowAssign[i] = j;
                                        colAssign[j] = i;
                                        done = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                });
        }

        return std::none_of(rowAssign.begin(), rowAssign.end(), 
                            [](unsigned int x) { return x == std::numeric_limits<unsigned int>::max(); });
    }

    void Hungarian::coverColumns(std::vector<std::vector<double>>& cost, 
                                 std::vector<unsigned int>& rowAssign, 
                                 std::vector<unsigned int>& colAssign, 
                                 std::vector<bool>& rowCovered, 
                                 std::vector<bool>& colCovered) {
        std::vector<bool> starred(cost.size(), false);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, cost.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    if (rowAssign[i] != std::numeric_limits<unsigned int>::max() && cost[i][rowAssign[i]] == 0) {
                        starred[i] = true;
                    }
                }
            });

        std::vector<bool> primed(cost.size(), false);
        bool changed;
        do {
            changed = false;
            for (size_t i = 0; i < cost.size(); ++i) {
                if (!rowCovered[i] && !starred[i]) {
                    for (size_t j = 0; j < cost[i].size(); ++j) {
                        if (cost[i][j] == 0 && !colCovered[j]) {
                            primed[i] = true;
                            if (rowAssign[i] == std::numeric_limits<unsigned int>::max()) {
                                // Augment path
                                size_t current_i = i;
                                while (primed[current_i]) {
                                    unsigned int j = rowAssign[current_i];
                                    rowAssign[current_i] = colAssign[j];
                                    colAssign[j] = current_i;
                                    current_i = colAssign[j];
                                }
                                std::fill(rowCovered.begin(), rowCovered.end(), false);
                                std::fill(colCovered.begin(), colCovered.end(), false);
                                return;
                            } else {
                                rowCovered[i] = true;
                                colCovered[rowAssign[i]] = false;
                                changed = true;
                            }
                            break;
                        }
                    }
                }
            }
        } while (changed);

        for (size_t i = 0; i < cost.size(); ++i) {
            rowCovered[i] = !rowCovered[i];
        }
    }

    void Hungarian::adjustUncovered(std::vector<std::vector<double>>& cost, 
                                    std::vector<bool>& rowCovered, 
                                    std::vector<bool>& colCovered) {
        double minUncovered = std::numeric_limits<double>::max();
        tbb::spin_mutex minMutex;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, cost.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                double localMin = std::numeric_limits<double>::max();
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    if (!rowCovered[i]) {
                        for (size_t j = 0; j < cost[i].size(); ++j) {
                            if (!colCovered[j]) {
                                localMin = std::min(localMin, cost[i][j]);
                            }
                        }
                    }
                }
                tbb::spin_mutex::scoped_lock lock(minMutex);
                minUncovered = std::min(minUncovered, localMin);
            });

        if (minUncovered == std::numeric_limits<double>::max()) return;

        tbb::parallel_for(tbb::blocked_range<size_t>(0, cost.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    for (size_t j = 0; j < cost[i].size(); ++j) {
                        if (rowCovered[i]) {
                            cost[i][j] += minUncovered;
                        }
                        if (!colCovered[j]) {
                            cost[i][j] -= minUncovered;
                        }
                    }
                }
            });
    }

}  // namespace slam