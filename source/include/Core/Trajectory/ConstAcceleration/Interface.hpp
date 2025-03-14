#pragma once

#include <Eigen/Core>
#include <memory>
#include <tbb/concurrent_hash_map.h>  // TBB concurrent hash map

#include "Core/Problem/CostTerm/WeightLeastSqCostTerm.hpp"
#include "Core/Problem/Problem.hpp"
#include "Core/Solver/Covariance.hpp"
#include "Core/Trajectory/ConstAcceleration/Variables.hpp"
#include "Core/Trajectory/Interface.hpp"
#include "Core/Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // Hash and equality comparator for Time
            struct TimeHashCompare {
                /**
                 * @brief Hash function for `Time` using nanoseconds.
                 * @param key The time key to hash.
                 * @return Hashed value.
                 */
                static size_t hash(const Time& key) {
                    int64_t nsec = key.nanosecs();
                    return std::hash<int64_t>{}(nsec) ^ (nsec >> 32);  // Mix higher bits
                }

                /**
                 * @brief Equality function for `Time`.
                 * @param a First time key.
                 * @param b Second time key.
                 * @return `true` if keys are equal.
                 */
                static bool equal(const Time& a, const Time& b) {
                    return a.nanosecs() == b.nanosecs();
                }
            };

            // -----------------------------------------------------------------------------
            /**
             * @class Interface
             * @brief Defines an interface for managing SE(3) trajectory states, priors, and interpolation.
             *
             * This class maintains state variables (pose, velocity, acceleration) in a trajectory and
             * provides methods for interpolation, covariance retrieval, and prior cost term management.
             */
            class Interface : public traj::Interface {
            public:
                using Ptr = std::shared_ptr<Interface>;
                using ConstPtr = std::shared_ptr<const Interface>;

                using PoseType = liemath::se3::Transformation;
                using VelocityType = Eigen::Matrix<double, 6, 1>;
                using AccelerationType = Eigen::Matrix<double, 6, 1>;
                using CovType = Eigen::Matrix<double, 18, 18>;

                // Define concurrent hash map type
                using KnotMap = tbb::concurrent_hash_map<Time, Variable::Ptr, TimeHashCompare>;

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
                /** @brief Adds a new state (pose, velocity, acceleration) to the trajectory. */
                void add(const Time& time, 
                         const slam::eval::Evaluable<PoseType>::Ptr& T_k0,
                         const slam::eval::Evaluable<VelocityType>::Ptr& w_0k_ink,
                         const slam::eval::Evaluable<AccelerationType>::Ptr& dw_0k_ink);

                // -----------------------------------------------------------------------------
                /** @brief Retrieves the state variable at a given time. */
                Variable::ConstPtr get(const Time& time) const;

                // -----------------------------------------------------------------------------
                /** @brief Retrieves interpolators for pose, velocity, and acceleration. */
                slam::eval::Evaluable<PoseType>::ConstPtr getPoseInterpolator(const Time& time) const;
                slam::eval::Evaluable<VelocityType>::ConstPtr getVelocityInterpolator(const Time& time) const;
                slam::eval::Evaluable<AccelerationType>::ConstPtr getAccelerationInterpolator(const Time& time) const;

                // -----------------------------------------------------------------------------
                /** @brief Retrieves process noise covariance at a given time. */
                CovType getCovariance(const slam::solver::Covariance& cov, const Time& time) const;

                // -----------------------------------------------------------------------------
                /** @brief Adds prior constraints for pose, velocity, acceleration, and full state. */
                void addPosePrior(const Time& time, const PoseType& T_k0, const Eigen::Matrix<double, 6, 6>& cov);
                void addVelocityPrior(const Time& time, const VelocityType& w_0k_ink, const Eigen::Matrix<double, 6, 6>& cov);
                void addAccelerationPrior(const Time& time, const AccelerationType& dw_0k_ink, const Eigen::Matrix<double, 6, 6>& cov);
                void addStatePrior(const Time& time, const PoseType& T_k0,
                                   const VelocityType& w_0k_ink, 
                                   const AccelerationType& dw_0k_ink, 
                                   const CovType& cov);

                // -----------------------------------------------------------------------------
                /** @brief Adds prior cost terms to the optimization problem. */
                void addPriorCostTerms(slam::problem::Problem& problem) const;

                // -----------------------------------------------------------------------------
                /** @brief Process noise diagonal. */
                Eigen::Matrix<double, 6, 1> Qc_diag_;

                // -----------------------------------------------------------------------------
                /** @brief Public access to covariance and transformation matrices. */
                Eigen::Matrix<double, 18, 18> getQinvPublic(const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const;
                Eigen::Matrix<double, 18, 18> getQPublic(const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const;
                Eigen::Matrix<double, 18, 18> getQinvPublic(const double& dt) const;
                Eigen::Matrix<double, 18, 18> getQPublic(const double& dt) const;
                Eigen::Matrix<double, 18, 18> getTranPublic(const double& dt) const;

            protected:
                // -----------------------------------------------------------------------------
                /** @brief Concurrent hash map storing trajectory knots. */
                KnotMap knot_map_;

                // -----------------------------------------------------------------------------
                /** @brief Weighted least-squares cost terms for pose, velocity, acceleration, and full state. */
                slam::problem::costterm::WeightedLeastSqCostTerm<6>::Ptr pose_prior_factor_ = nullptr;
                slam::problem::costterm::WeightedLeastSqCostTerm<6>::Ptr vel_prior_factor_ = nullptr;
                slam::problem::costterm::WeightedLeastSqCostTerm<6>::Ptr acc_prior_factor_ = nullptr;
                slam::problem::costterm::WeightedLeastSqCostTerm<18>::Ptr state_prior_factor_ = nullptr;

                // -----------------------------------------------------------------------------
                /** @brief Internal methods for Jacobian calculations. */
                Eigen::Matrix<double, 18, 18> getJacKnot1_(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const;
                Eigen::Matrix<double, 18, 18> getJacKnot2_(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const;

                // -----------------------------------------------------------------------------
                /** @brief Internal process noise covariance computations. */
                Eigen::Matrix<double, 18, 18> getQ_(const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const;
                Eigen::Matrix<double, 18, 18> getQinv_(const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const;

                // -----------------------------------------------------------------------------
                /** @brief Internal methods for interpolators. */
                slam::eval::Evaluable<PoseType>::Ptr getPoseInterpolator_(const Time& time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const;
                slam::eval::Evaluable<VelocityType>::Ptr getVelocityInterpolator_(const Time& time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const;
                slam::eval::Evaluable<AccelerationType>::Ptr getAccelerationInterpolator_(const Time& time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const;

                // -----------------------------------------------------------------------------
                /** @brief Internal methods for extrapolators. */
                slam::eval::Evaluable<PoseType>::Ptr getPoseExtrapolator_(const Time& time, const Variable::ConstPtr& knot) const;
                slam::eval::Evaluable<VelocityType>::Ptr getVelocityExtrapolator_(const Time& time, const Variable::ConstPtr& knot) const;
                slam::eval::Evaluable<AccelerationType>::Ptr getAccelerationExtrapolator_(const Time& time, const Variable::ConstPtr& knot) const;

                // -----------------------------------------------------------------------------
                /** @brief Internal method to compute prior factor. */
                slam::eval::Evaluable<Eigen::Matrix<double, 18, 1>>::Ptr getPriorFactor_(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const;
            };
        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam