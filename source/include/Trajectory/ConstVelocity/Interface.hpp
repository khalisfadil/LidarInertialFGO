#pragma once

#include <Eigen/Core>
#include <memory>

#include "source/include/Problem/CostTerm/WeightLeastSqCostTerm.hpp"
#include "source/include/Problem/Problem.hpp"
#include "source/include/Solver/Covariance.hpp"
#include "source/include/Trajectory/ConstVelocity/Variables.hpp"
#include "source/include/Trajectory/Interface.hpp"
#include "source/include/Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace const_vel {

            // -----------------------------------------------------------------------------
            /**
             * @class Interface
             * @brief Manages SE(3) trajectory states, priors, and interpolation.
             *
             * This class maintains a trajectory representation in SE(3), allowing
             * pose and velocity interpolation, covariance retrieval, and factor graph
             * optimization with prior cost terms.
             */
            class Interface : public traj::Interface {
                public:
                    using Ptr = std::shared_ptr<Interface>;
                    using ConstPtr = std::shared_ptr<const Interface>;

                    using PoseType = liemath::se3::Transformation;
                    using VelocityType = Eigen::Matrix<double, 6, 1>;
                    using CovType = Eigen::Matrix<double, 12, 12>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create an instance of `Interface`.
                     * @param Qc_diag Process noise diagonal (default: ones).
                     * @return Shared pointer to the created instance.
                     */
                    static Ptr MakeShared(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& Qc_diag = Eigen::Matrix<double, 6, 1>::Ones());

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructs an `Interface` instance.
                     * @param Qc_diag Process noise diagonal (default: ones).
                     */
                    explicit Interface(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& Qc_diag = Eigen::Matrix<double, 6, 1>::Ones());

                    // -----------------------------------------------------------------------------
                    /** @brief Adds a new state (pose, velocity) to the trajectory. */
                    void add(const Time& time, 
                            const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                            const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink);

                    // -----------------------------------------------------------------------------
                    /** @brief Retrieves the state variable at a given time. */
                    Variable::ConstPtr get(const Time& time) const;

                    // -----------------------------------------------------------------------------
                    /** @brief Retrieves interpolators for pose and velocity. */
                    slam::eval::Evaluable<PoseType>::ConstPtr getPoseInterpolator(const Time& time) const;
                    slam::eval::Evaluable<VelocityType>::ConstPtr getVelocityInterpolator(const Time& time) const;

                    // -----------------------------------------------------------------------------
                    /** @brief Retrieves process noise covariance at a given time. */
                    CovType getCovariance(const slam::solver::Covariance& cov, const Time& time);

                    // -----------------------------------------------------------------------------
                    /** @brief Adds prior constraints for pose, velocity, and full state. */
                    void addPosePrior(const Time& time, const PoseType& T_k0, const Eigen::Matrix<double, 6, 6>& cov);
                    void addVelocityPrior(const Time& time, const VelocityType& w_0k_ink, const Eigen::Matrix<double, 6, 6>& cov);
                    void addStatePrior(const Time& time, const PoseType& T_k0,
                                    const VelocityType& w_0k_ink, 
                                    const CovType& cov);

                    // -----------------------------------------------------------------------------
                    /** @brief Adds prior cost terms to the optimization problem. */
                    void addPriorCostTerms(slam::problem::Problem& problem) const;

                private:

                    // -----------------------------------------------------------------------------
                    /** @brief Process noise diagonal. */
                    Eigen::Matrix<double, 6, 1> Qc_diag_;

                    // -----------------------------------------------------------------------------
                    /** @brief Map storing trajectory knots. */
                    std::map<Time, Variable::Ptr> knot_map_;

                    // -----------------------------------------------------------------------------
                    /** @brief Weighted least-squares cost terms for pose, velocity, and full state. */
                    slam::problem::costterm::WeightedLeastSqCostTerm<6>::Ptr pose_prior_factor_ = nullptr;
                    slam::problem::costterm::WeightedLeastSqCostTerm<6>::Ptr vel_prior_factor_ = nullptr;
                    slam::problem::costterm::WeightedLeastSqCostTerm<12>::Ptr state_prior_factor_ = nullptr;
            };

        }  // namespace const_vel
    }  // namespace traj
}  // namespace slam
