#pragma once

#include <Eigen/Core>
#include <memory>

#include "Evaluable/Evaluable.hpp"
#include "Evaluable/vspace/Evaluables.hpp"
#include "Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace bspline {
            
            // -----------------------------------------------------------------------------
            /**
             * @class Variable
             * @brief Represents a trajectory control point in a B-spline formulation.
             *
             * This class stores a timestamped **control vector** (c), which defines the
             * trajectory's local behavior in an optimization problem.
             */
            class Variable {
                public:
                    using Ptr = std::shared_ptr<Variable>;
                    using ConstPtr = std::shared_ptr<const Variable>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared instance of Variable.
                     * @param time Timestamp of the control point.
                     * @param c Control vector representing the state.
                     * @return Shared pointer to the created Variable instance.
                     */
                    static Ptr MakeShared(const slam::traj::Time& time,
                                          const slam::eval::vspace::VSpaceStateVar<6>::Ptr& c) {
                        return std::make_shared<Variable>(time, c);
                    }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructs a Variable.
                     * @param time Timestamp of the control point.
                     * @param c Control vector representing the state.
                     */
                    explicit Variable(const slam::traj::Time& time,
                                      const slam::eval::vspace::VSpaceStateVar<6>::Ptr& c)
                        : time_(time), c_(std::move(c)) {}

                    /** @brief Default destructor */
                    ~Variable() = default;
                    
                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Get the timestamp of this variable.
                     * @return Reference to the stored timestamp.
                     */
                    const slam::traj::Time& getTime() const { return time_; }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Get the control vector associated with this variable.
                     * @return Shared pointer to the control vector.
                     */
                    const slam::eval::vspace::VSpaceStateVar<6>::Ptr& getC() const { return c_; }

                private:
                    // -----------------------------------------------------------------------------
                    /** @brief Timestamp of the control point */
                    slam::traj::Time time_;

                    // -----------------------------------------------------------------------------
                    /** @brief Control vector representing the state */
                    slam::eval::vspace::VSpaceStateVar<6>::Ptr c_;
            };

        }  // namespace bspline
    }  // namespace traj
}  // namespace steam
