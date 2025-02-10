#include "source/include/Trajectory/Bspline/Interface.hpp"

#include "source/include/Evaluable/vspace/Evaluables.hpp"
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

                // Compute neighboring knots
                int64_t t2_nano = knot_spacing_.nanosecs() * 
                                  std::floor(time.nanosecs() / knot_spacing_.nanosecs());
                slam::traj::Time t2(t2_nano);
                slam::traj::Time t1 = t2 - knot_spacing_;
                slam::traj::Time t3 = t2 + knot_spacing_;
                slam::traj::Time t4 = t3 + knot_spacing_;

                // Efficient knot insertion using try_emplace
                const auto v1 = knot_map_.try_emplace(
                    t1, slam::traj::bspline::Variable::MakeShared(
                        t1, slam::eval::vspace::VSpaceStateVar<6>::MakeShared(VeloType::Zero())
                    )).first->second;

                const auto v2 = knot_map_.try_emplace(
                    t2, slam::traj::bspline::Variable::MakeShared(
                        t2, slam::eval::vspace::VSpaceStateVar<6>::MakeShared(VeloType::Zero())
                    )).first->second;

                const auto v3 = knot_map_.try_emplace(
                    t3, slam::traj::bspline::Variable::MakeShared(
                        t3, slam::eval::vspace::VSpaceStateVar<6>::MakeShared(VeloType::Zero())
                    )).first->second;

                const auto v4 = knot_map_.try_emplace(
                    t4, slam::traj::bspline::Variable::MakeShared(
                        t4, slam::eval::vspace::VSpaceStateVar<6>::MakeShared(VeloType::Zero())
                    )).first->second;

                return slam::traj::bspline::VelocityInterpolator::MakeShared(time, v1, v2, v3, v4);
            }

        }  // namespace bspline
    }  // namespace traj
}  // namespace slam
