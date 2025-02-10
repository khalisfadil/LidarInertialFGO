#pragma once

#include <memory>

#include "source/include/Problem/OptimizationProblem.hpp"

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
            /// Shared pointer type aliases for better readability.
            using Ptr = std::shared_ptr<Interface>;
            using ConstPtr = std::shared_ptr<const Interface>;

            // -----------------------------------------------------------------------------
            /// @brief Virtual destructor to ensure proper cleanup in derived classes.
            virtual ~Interface() = default;

            // -----------------------------------------------------------------------------
            /**
             * @brief Returns a reference to the associated optimization problem.
             * @return Shared pointer to the underlying optimization problem.
             */
            virtual slam::problem::OptimizationProblem::Ptr getOptimizationProblem() const = 0;
        };

    }  // namespace traj
}  // namespace slam
