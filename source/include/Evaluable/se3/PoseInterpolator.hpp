#pragma once

#include <Eigen/Core>
#include <array>

#include "source/include/LGMath/LieGroupMath.hpp"
#include "source/include/Evaluable/Evaluable.hpp"
#include "source/include/Trajectory/Time.hpp"

namespace slam {
    namespace eval {
        namespace se3 {

            // -----------------------------------------------------------------------------
            /**
             * @class PoseInterpolator
             * @brief SE(3) pose interpolation using Lie group logarithm and exponential maps.
             *
             * Given two transformations \(T_1\) and \(T_2\) at different timestamps \(t_1\) and \(t_2\),
             * this class interpolates a transformation \(T_i\) at time \(t\) using:
             *
             * \f[
             * T_i = \exp(\alpha \cdot \log(T_2 T_1^{-1})) T_1
             * \f]
             *
             * where:
             * - \( \alpha = \frac{t - t_1}{t_2 - t_1} \) is the interpolation factor.
             * - Faulhaber series is used for higher-order corrections.
             *
             * This class is **optimized for real-time computation**, reducing memory allocations and improving efficiency.
             */
            class PoseInterpolator : public Evaluable<slam::liemath::se3::Transformation> {
            public:
                using Ptr = std::shared_ptr<PoseInterpolator>;
                using ConstPtr = std::shared_ptr<const PoseInterpolator>;
                using Time = slam::traj::Time;

                using InType = slam::liemath::se3::Transformation;
                using OutType = slam::liemath::se3::Transformation;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create a shared instance of PoseInterpolator.
                 *
                 * @param time The query time for interpolation.
                 * @param transform1 First transformation \(T_1\).
                 * @param time1 Timestamp of \(T_1\).
                 * @param transform2 Second transformation \(T_2\).
                 * @param time2 Timestamp of \(T_2\).
                 * @return Shared pointer to a new PoseInterpolator instance.
                 */
                static Ptr MakeShared(const Time& time,
                                      const Evaluable<InType>::ConstPtr& transform1,
                                      const Time& time1,
                                      const Evaluable<InType>::ConstPtr& transform2,
                                      const Time& time2);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs a PoseInterpolator instance.
                 *
                 * @param time The query time for interpolation.
                 * @param transform1 First transformation \(T_1\).
                 * @param time1 Timestamp of \(T_1\).
                 * @param transform2 Second transformation \(T_2\).
                 * @param time2 Timestamp of \(T_2\).
                 */
                PoseInterpolator(const Time& time,
                                 const Evaluable<InType>::ConstPtr& transform1,
                                 const Time& time1,
                                 const Evaluable<InType>::ConstPtr& transform2,
                                 const Time& time2);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Checks if either of the input transformations is active.
                 * @return True if at least one transformation is active, otherwise false.
                 */
                bool active() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves related state keys.
                 * @param[out] keys The set of state keys related to this evaluator.
                 */
                void getRelatedVarKeys(KeySet& keys) const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes the interpolated transformation \(T_i\).
                 * @return The interpolated SE(3) transformation.
                 */
                OutType value() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Forward pass: Computes the interpolated transformation and stores it in a node.
                 * @return A node containing the computed transformation \(T_i\).
                 */
                Node<OutType>::Ptr forward() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Backward pass: Computes Jacobians for optimization.
                 *
                 * @param lhs Left-hand-side weight matrix.
                 * @param node Node containing the forward-pass result.
                 * @param jacs Container to store the computed Jacobians.
                 */
                void backward(const Eigen::Ref<const Eigen::MatrixXd>& lhs, 
                    const Node<OutType>::Ptr& node, 
                    StateKeyJacobians& jacs) const override;


            private:
                const Evaluable<InType>::ConstPtr transform1_;  ///< First transformation \(T_1\).
                const Evaluable<InType>::ConstPtr transform2_;  ///< Second transformation \(T_2\).
                double alpha_;  ///< Interpolation factor \( \alpha \).
                std::array<double, 4> faulhaber_coeffs_;  ///< **Precomputed Faulhaber coefficients**.
            };

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
