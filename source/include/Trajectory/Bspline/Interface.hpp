#pragma once

#include <Eigen/Core>
#include <memory>
#include <map>

#include "source/include/Problem/CostTerm/WeightLeastSqCostTerm.hpp"
#include "source/include/Problem/OptimizationProblem.hpp"
#include "source/include/Trajectory/Bspline/Variable.hpp"
#include "source/include/Trajectory/Interface.hpp"
#include "source/include/Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace bspline {

            // -----------------------------------------------------------------------------
            /**
             * @class Interface
             * @brief Provides B-spline trajectory functionality and knot management.
             *
             * This class manages B-spline trajectory knots, provides velocity interpolators,
             * and integrates with the optimization framework.
             */
            class Interface : public slam::traj::Interface {
            public:
                using Ptr = std::shared_ptr<Interface>;
                using ConstPtr = std::shared_ptr<const Interface>;

                using VeloType = Eigen::Matrix<double, 6, 1>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create a shared instance of Interface.
                 * @param knot_spacing Interval between knots.
                 * @return Shared pointer to the created Interface instance.
                 */
                static Ptr MakeShared(const slam::traj::Time& knot_spacing = slam::traj::Time(0.1));

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs a B-spline interface.
                 * @param knot_spacing Time interval between trajectory knots.
                 */
                explicit Interface(const slam::traj::Time& knot_spacing = slam::traj::Time(0.1));

                // -----------------------------------------------------------------------------
                /**
                 * @brief Get velocity interpolator at a specific time.
                 * @param time The query time.
                 * @return ConstPtr to an evaluable velocity interpolator.
                 */
                slam::eval::Evaluable<VeloType>::ConstPtr getVelocityInterpolator(const slam::traj::Time& time);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Adds all state variables to the optimization problem.
                 * @param problem The optimization problem to update.
                 */
                void addStateVariables(slam::problem::OptimizationProblem& problem) const;

                // -----------------------------------------------------------------------------
                /** @brief Retrieves the internal knot map. */
                using KnotMap = std::map<slam::traj::Time, slam::traj::bspline::Variable::Ptr>;
                const KnotMap& getKnotMap() const { return knot_map_; }

            protected:
                /** @brief Spacing between trajectory knots */
                const slam::traj::Time knot_spacing_;

                /** @brief Ordered map of trajectory knots */
                KnotMap knot_map_;
            };

        }  // namespace bspline
    }  // namespace traj
}  // namespace slam
