#pragma once

#include <memory>

#include "Problem/OptimizationProblem.hpp"

namespace slam {
    namespace traj {
        
        // -----------------------------------------------------------------------------
        /**
         * @class Interface
         * @brief Abstract base class for trajectory representations in SLAM.
         *
         * This interface serves as a **base class for trajectory optimization modules**.
         * 
         * Possible applications:
         * - **Pose Graph Optimization (PGO)**
         * - **Spline-based trajectory representations**
         * - **Factor graph-based trajectory estimation**
         */
        class Interface {
            public:
            // -----------------------------------------------------------------------------
            /// Shared pointer type aliases for readability.
            using Ptr = std::shared_ptr<Interface>;
            using ConstPtr = std::shared_ptr<const Interface>;

            // -----------------------------------------------------------------------------
            /// @brief Virtual destructor to ensure proper cleanup in derived classes.
            virtual ~Interface() = default;

            // -----------------------------------------------------------------------------
            // No pure virtual function here, no need for `getOptimizationProblem()`.
        };

    }  // namespace traj
}  // namespace slam
