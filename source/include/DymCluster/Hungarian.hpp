#pragma once

#include <vector>
#include <limits>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/spin_mutex.h>

namespace slam {

    class Hungarian {
    public:
        // Updated to return std::vector<unsigned int> for assignment
        static double solve(const std::vector<std::vector<double>>& costMatrix, 
                            std::vector<unsigned int>& assignment);

    private:
        static void subtractRowMinima(std::vector<std::vector<double>>& cost);
        static void subtractColMinima(std::vector<std::vector<double>>& cost);

        // Updated rowAssign and colAssign to std::vector<unsigned int>
        static bool findAssignment(std::vector<std::vector<double>>& cost, 
                                   std::vector<unsigned int>& rowAssign, 
                                   std::vector<unsigned int>& colAssign, 
                                   std::vector<bool>& rowCovered, 
                                   std::vector<bool>& colCovered);

        // Updated rowAssign and colAssign to std::vector<unsigned int>
        static void coverColumns(std::vector<std::vector<double>>& cost, 
                                 std::vector<unsigned int>& rowAssign, 
                                 std::vector<unsigned int>& colAssign, 
                                 std::vector<bool>& rowCovered, 
                                 std::vector<bool>& colCovered);

        static void adjustUncovered(std::vector<std::vector<double>>& cost, 
                                    std::vector<bool>& rowCovered, 
                                    std::vector<bool>& colCovered);
    };

}  // namespace slam