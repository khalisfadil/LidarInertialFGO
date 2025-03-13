#pragma once

#include <Eigen/Core>
#include <memory>

#include "Evaluable/Evaluable.hpp"
#include "Trajectory/Time.hpp"
#include "Trajectory/Bspline/Variable.hpp"

namespace slam {
    namespace traj {
        namespace bspline {

            // -----------------------------------------------------------------------------
            /**
             * @class VelocityInterpolator
             * @brief Interpolates velocity using a cubic B-spline formulation.
             *
             * This class computes an interpolated velocity at a given timestamp using
             * four trajectory control points (`k1, k2, k3, k4`). The B-spline basis
             * function weights are precomputed for efficiency.
             */
            class VelocityInterpolator : public slam::eval::Evaluable<Eigen::Matrix<double, 6, 1>> {
                public:
                    using Ptr = std::shared_ptr<VelocityInterpolator>;
                    using ConstPtr = std::shared_ptr<const VelocityInterpolator>;

                    using CType = Eigen::Matrix<double, 6, 1>;
                    using OutType = Eigen::Matrix<double, 6, 1>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared instance of VelocityInterpolator.
                     * @param time Time at which velocity is interpolated.
                     * @param k1 First control point.
                     * @param k2 Second control point.
                     * @param k3 Third control point.
                     * @param k4 Fourth control point.
                     * @return Shared pointer to created VelocityInterpolator instance.
                     */
                    static Ptr MakeShared(const slam::traj::Time& time,
                                          const slam::traj::bspline::Variable::ConstPtr& k1,
                                          const slam::traj::bspline::Variable::ConstPtr& k2,
                                          const slam::traj::bspline::Variable::ConstPtr& k3,
                                          const slam::traj::bspline::Variable::ConstPtr& k4);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor for VelocityInterpolator.
                     * @param time Time at which velocity is interpolated.
                     * @param k1 First control point.
                     * @param k2 Second control point.
                     * @param k3 Third control point.
                     * @param k4 Fourth control point.
                     */
                    explicit VelocityInterpolator(const slam::traj::Time& time,
                                                 const slam::traj::bspline::Variable::ConstPtr& k1,
                                                 const slam::traj::bspline::Variable::ConstPtr& k2,
                                                 const slam::traj::bspline::Variable::ConstPtr& k3,
                                                 const slam::traj::bspline::Variable::ConstPtr& k4);

                    // -----------------------------------------------------------------------------
                    /** @brief Checks if the interpolator is active (any control point is active). */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Retrieves keys of related variables. */
                    void getRelatedVarKeys(KeySet& keys) const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Computes the interpolated velocity value. */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Computes the forward evaluation of velocity. */
                    slam::eval::Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Computes the backward propagation of Jacobians. */
                    void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs, 
                                  const slam::eval::Node<OutType>::Ptr& node, 
                                  slam::eval::StateKeyJacobians& jacs) const override;

                private:

                    // -----------------------------------------------------------------------------
                    /** @brief Control points for interpolation */
                    const slam::traj::bspline::Variable::ConstPtr k1_, k2_, k3_, k4_;

                    // -----------------------------------------------------------------------------
                    /** @brief B-spline weights for interpolation */
                    Eigen::Matrix<double, 4, 1> w_;

                    // -----------------------------------------------------------------------------
                    /** @brief B-spline basis matrix (precomputed) */
                    static const Eigen::Matrix4d B;
            };

        }  // namespace bspline
    }  // namespace traj
}  // namespace slam
