#pragma once

#include <Eigen/Core>

#include "source/include/Evaluable/Evaluable.hpp"
#include "source/include/Trajectory/ConstAcceleration/Variables.hpp"
#include "source/include/Trajectory/Time.hpp"

namespace slam {
    namespace traj {
        namespace const_acc {

            // -----------------------------------------------------------------------------
            /**
             * @class PoseExtrapolator
             * @brief SE(3) Pose Extrapolation for a Constant Acceleration Model.
             *
             * This class performs **pose extrapolation** given an initial knot state.
             * It predicts the **SE(3) pose** at a future time using a constant acceleration model.
             */
            class PoseExtrapolator : public eval::Evaluable<liemath::se3::Transformation> {
                public:
                    using Ptr = std::shared_ptr<PoseExtrapolator>;
                    using ConstPtr = std::shared_ptr<const PoseExtrapolator>;

                    using InPoseType = liemath::se3::Transformation;
                    using InVelType = Eigen::Matrix<double, 6, 1>;
                    using InAccType = Eigen::Matrix<double, 6, 1>;
                    using OutType = liemath::se3::Transformation;

                    // -----------------------------------------------------------------------------
                    /** @brief Factory method */
                    static Ptr MakeShared(const Time& time, const Variable::ConstPtr& knot);

                    // -----------------------------------------------------------------------------
                    /** @brief Constructor */
                    PoseExtrapolator(const Time& time, const Variable::ConstPtr& knot);

                    // -----------------------------------------------------------------------------
                    /** @brief Check if any dependent variables are active */
                    bool active() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Collect related variable keys */
                    void getRelatedVarKeys(KeySet& keys) const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Compute extrapolated pose value */
                    OutType value() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Forward evaluation for the extrapolated pose */
                    eval::Node<OutType>::Ptr forward() const override;

                    // -----------------------------------------------------------------------------
                    /** @brief Backpropagation step for optimization */
                    void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs,
                              const eval::Node<OutType>::Ptr& node,
                              eval::StateKeyJacobians& jacs) const override;

                protected:

                    // -----------------------------------------------------------------------------
                    /** @brief Knot to extrapolate from */
                    const Variable::ConstPtr knot_;

                    // -----------------------------------------------------------------------------
                    /** @brief Transition matrix for extrapolation */
                    Eigen::Matrix<double, 18, 18> Phi_;
            };
        }  // namespace const_acc
    }  // namespace traj
}  // namespace slam
