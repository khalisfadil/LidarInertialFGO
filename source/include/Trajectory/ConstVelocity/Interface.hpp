#pragma once

#include <Eigen/Core>
#include <memory>
#include <map>

#include "source/include/Problem/CostTerm/WeightLeastSqCostTerm.hpp"
#include "source/include/Problem/Problem.hpp"
#include "source/include/Solver/Covariance.hpp"
#include "source/include/Trajectory/ConstVelocity/Variables.hpp"
#include "source/include/Trajectory/Interface.hpp"
#include "source/include/Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {

            /**
             * @class Interface
             * @brief Implements a constant velocity trajectory model.
             *
             * Provides:
             * - **State interpolation** (pose & velocity)
             * - **Covariance propagation**
             * - **Prior constraints** for factor graph optimization
             */
            class Interface : public traj::Interface {
            public:
                using Ptr = std::shared_ptr<Interface>;
                using ConstPtr = std::shared_ptr<const Interface>;

                using PoseType = slam::liemath::se3::Transformation;
                using VelocityType = Eigen::Matrix<double, 6, 1>;
                using CovType = Eigen::Matrix<double, 12, 12>;

                /**
                 * @brief Factory method to create an instance.
                 */
                static Ptr MakeShared(
                    const Eigen::Matrix<double, 6, 1>& Qc_diag = Eigen::Matrix<double, 6, 1>::Ones());

                /**
                 * @brief Constructor for the Interface.
                 */
                explicit Interface(
                    const Eigen::Matrix<double, 6, 1>& Qc_diag = Eigen::Matrix<double, 6, 1>::Ones());

                /**
                 * @brief Virtual destructor (ensures proper cleanup in derived classes).
                 */
                ~Interface() override = default;

                void add(const slam::traj::Time& time,
                         const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                         const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink);

                Variable::ConstPtr get(const slam::traj::Time& time) const;

                slam::eval::Evaluable<PoseType>::ConstPtr getPoseInterpolator(const slam::traj::Time& time) const;

                slam::eval::Evaluable<VelocityType>::ConstPtr getVelocityInterpolator(const slam::traj::Time& time) const;

                CovType getCovariance(const slam::solver::Covariance& cov, const slam::traj::Time& time);

                void addPosePrior(const slam::traj::Time& time, const PoseType& T_k0, const Eigen::Matrix<double, 6, 6>& cov);
                void addVelocityPrior(const slam::traj::Time& time, const VelocityType& w_0k_ink, const Eigen::Matrix<double, 6, 6>& cov);
                void addStatePrior(const slam::traj::Time& time, const PoseType& T_k0, const VelocityType& w_0k_ink, const CovType& cov);

                void addPriorCostTerms(slam::problem::Problem& problem) const;

            private:
                Eigen::Matrix<double, 6, 1> Qc_diag_;
                std::map<slam::traj::Time, Variable::Ptr> knot_map_;

                slam::problem::costterm::WeightedLeastSqCostTerm<6>::Ptr pose_prior_factor_ = nullptr;
                slam::problem::costterm::WeightedLeastSqCostTerm<6>::Ptr vel_prior_factor_ = nullptr;
                slam::problem::costterm::WeightedLeastSqCostTerm<12>::Ptr state_prior_factor_ = nullptr;
            };

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
