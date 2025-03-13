#pragma once

#include <Eigen/Core>
#include <memory>

#include "Trajectory/ConstAcceleration/Interface.hpp"
#include "Trajectory/Time.hpp"

#include "Trajectory/Singer/AccelerationExtrapolator.hpp"
#include "Trajectory/Singer/AccelerationInterpolator.hpp"
#include "Trajectory/Singer/Helper.hpp"
#include "Trajectory/Singer/PoseExtrapolator.hpp"
#include "Trajectory/Singer/PoseInterpolator.hpp"
#include "Trajectory/Singer/PriorFactor.hpp"
#include "Trajectory/Singer/VelocityExtrapolator.hpp"
#include "Trajectory/Singer/VelocityInterpolator.hpp"

namespace slam {
    namespace traj {
        namespace singer {

            // -----------------------------------------------------------------------------
            /**
             * @class Interface
             * @brief Manages SE(3) trajectory states, priors, and interpolations using **Singer motion model**.
             *
             * This class extends the **constant-acceleration motion model** by introducing **damping effects**,
             * improving motion continuity and stability. Supports interpolation, covariance retrieval,  
             * and prior cost term management in **Gaussian Process (GP) Trajectory Estimation**.
             */
            class Interface : public slam::traj::const_acc::Interface {
            public:
                using Ptr = std::shared_ptr<Interface>;
                using ConstPtr = std::shared_ptr<const Interface>;
                using Variable = slam::traj::const_acc::Variable;

                using PoseType = liemath::se3::Transformation;
                using VelocityType = Eigen::Matrix<double, 6, 1>;
                using AccelerationType = Eigen::Matrix<double, 6, 1>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create an instance of `Interface`.
                 * @param alpha_diag Damping coefficient vector (default: ones).
                 * @param Qc_diag Process noise diagonal (default: ones).
                 * @return Shared pointer to the created instance.
                 */
                static Ptr MakeShared(const Eigen::Matrix<double, 6, 1>& alpha_diag = Eigen::Matrix<double, 6, 1>::Ones(),
                                    const Eigen::Matrix<double, 6, 1>& Qc_diag = Eigen::Matrix<double, 6, 1>::Ones());

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs an `Interface` instance.
                 * @param alpha_diag Damping coefficient vector (default: ones).
                 * @param Qc_diag Process noise diagonal (default: ones).
                 */
                explicit Interface(const Eigen::Matrix<double, 6, 1>& alpha_diag = Eigen::Matrix<double, 6, 1>::Ones(),
                                const Eigen::Matrix<double, 6, 1>& Qc_diag = Eigen::Matrix<double, 6, 1>::Ones());

                // -----------------------------------------------------------------------------
                /** @brief Public access to covariance and transformation matrices. */
                Eigen::Matrix<double, 18, 18> getQinvPublic(const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const;
                Eigen::Matrix<double, 18, 18> getQPublic(const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const;
                Eigen::Matrix<double, 18, 18> getQinvPublic(const double& dt) const;
                Eigen::Matrix<double, 18, 18> getQPublic(const double& dt) const;
                Eigen::Matrix<double, 18, 18> getTranPublic(const double& dt) const;

            protected:
                Eigen::Matrix<double, 6, 1> alpha_diag_;  ///< Damping coefficient vector

                // -----------------------------------------------------------------------------
                /** @brief Jacobian calculation. */
                Eigen::Matrix<double, 18, 18> getJacKnot1_(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const;

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

        }  // namespace singer
    }  // namespace traj
}  // namespace slam
