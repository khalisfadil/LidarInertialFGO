#include "source/include/Trajectory/Bspline/Interface.hpp"

#include "source/include/Evaluable/se3/Evaluables.hpp"
#include "source/include/Evaluable/vspace/Evaluables.hpp"
#include "source/include/Problem/LossFunc/LossFunc.hpp"
#include "source/include/Problem/NoiseModel/StaticNoiseModel.hpp"
#include "source/include/Trajectory/Bspline/VelocityInterpolator.hpp"

namespace slam {
    namespace traj {
        namespace bspline {
            
            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto Interface::MakeShared(const slam::traj::Time& knot_spacing) -> Ptr {
                return std::make_shared<Interface>(knot_spacing);
            }

            // -----------------------------------------------------------------------------
            // Interface
            // -----------------------------------------------------------------------------

            Interface::Interface(const slam::traj::Time& knot_spacing)
                : knot_spacing_(knot_spacing) {}

            // -----------------------------------------------------------------------------
            // addStateVariables
            // -----------------------------------------------------------------------------

            void Interface::addStateVariables(slam::problem::OptimizationProblem& problem) const {
                for (const auto& [time, variable] : knot_map_) {
                    problem.addStateVariable(variable->getC());
                }
            }

            // -----------------------------------------------------------------------------
            // getVelocityInterpolator
            // -----------------------------------------------------------------------------

            auto Interface::getVelocityInterpolator(const slam::traj::Time& time) 
                -> slam::eval::Evaluable<VeloType>::ConstPtr {
                
                // Compute relevant time knots
                const int64_t t2_nano = knot_spacing_.nanosecs() *
                                        std::floor(time.nanosecs() / knot_spacing_.nanosecs());
                const Time t2(t2_nano);
                const Time t1 = t2 - knot_spacing_;
                const Time t3 = t2 + knot_spacing_;
                const Time t4 = t3 + knot_spacing_;

                // Function to create or retrieve a velocity state variable
                auto get_or_create_knot = [&](const Time& t) -> Variable::ConstPtr {
                    return knot_map_.try_emplace(
                        t, Variable::MakeShared(t, slam::eval::vspace::VSpaceStateVar<6>::MakeShared(Eigen::Matrix<double, 6, 1>::Zero()))
                    ).first->second;
                };

                // Retrieve or create required knots
                const auto v1 = get_or_create_knot(t1);
                const auto v2 = get_or_create_knot(t2);
                const auto v3 = get_or_create_knot(t3);
                const auto v4 = get_or_create_knot(t4);

                return VelocityInterpolator::MakeShared(time, v1, v2, v3, v4);
            }
        }  // namespace bspline
    }  // namespace traj
}  // namespace slam
