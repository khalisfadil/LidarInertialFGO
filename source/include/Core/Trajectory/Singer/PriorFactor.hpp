#pragma once

#include <Eigen/Core>
#include <memory>

#include "Core/Trajectory/ConstAcceleration/PriorFactor.hpp"
#include "Core/Trajectory/ConstAcceleration/Variables.hpp"
#include "Core/Trajectory/Singer/Helper.hpp"
#include "Core/Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace singer {
            
            // -----------------------------------------------------------------------------
            /**
             * @class PriorFactor
             * @brief Implements a **generalized prior factor** for the Singer acceleration model.
             *
             * This class computes **Gaussian Process (GP) prior constraints** between  
             * two trajectory knots (`knot1`, `knot2`) in **SE(3) Lie Algebra**.
             *
             * **Key Features:**
             * - Supports **constant acceleration motion models**.
             * - Computes **state transition Jacobians**.
             * - Encapsulates **damping behavior** via `alpha_diag_`.
             */
            class PriorFactor : public slam::traj::const_acc::PriorFactor {
            public:
                using Ptr = std::shared_ptr<PriorFactor>;
                using ConstPtr = std::shared_ptr<const PriorFactor>;
                using Variable = slam::traj::const_acc::Variable;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create a shared instance of `PriorFactor`.
                 *
                 * @param time  Time at which the factor is evaluated.
                 * @param knot1 First trajectory knot (earlier state).
                 * @param knot2 Second trajectory knot (later state).
                 * @param ad    Damping coefficient vector (6x1) controlling acceleration behavior.
                 * @return      Shared pointer to a `PriorFactor` instance.
                 */
                static Ptr MakeShared(const Variable::ConstPtr& knot1,
                                      const Variable::ConstPtr& knot2,
                                      const Eigen::Matrix<double, 6, 1>& ad);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs the `PriorFactor` for **Singer acceleration motion model**.
                 *
                 * @param time  Time at which the factor is evaluated.
                 * @param knot1 First trajectory knot (earlier state).
                 * @param knot2 Second trajectory knot (later state).
                 * @param ad    Damping coefficient vector (6x1) affecting acceleration dynamics.
                 */
                PriorFactor(const Variable::ConstPtr& knot1,
                            const Variable::ConstPtr& knot2,
                            const Eigen::Matrix<double, 6, 1>& ad);

            protected:
                const Eigen::Matrix<double, 6, 1> alpha_diag_;  ///< **Damping coefficients** for acceleration.

                /**
                 * @brief Computes the **Jacobian of the state transition** with respect to `knot1`.
                 * @return 18x18 Jacobian matrix.
                 */
                Eigen::Matrix<double, 18, 18> getJacKnot1_() const;
            };

        }  // namespace singer
    }  // namespace traj
}  // namespace slam
