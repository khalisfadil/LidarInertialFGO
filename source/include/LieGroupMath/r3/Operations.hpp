
#pragma once

#include <stdexcept>
#include <Eigen/Dense>

#include <include/lieGroupMath/r3/Types.hpp>
#include <include/lieGroupMath/se3/Operations.hpp>
#include <include/lieGroupMath/se3/TransformationWithCovariance.hpp>

namespace slam {
    namespace liemath {
        namespace r3 {
            
            // -----------------------------------------------------------------------------
            /** \brief The transform covariance is required to be set */
            static constexpr bool COVARIANCE_REQUIRED = true;

            // -----------------------------------------------------------------------------
            /** \brief The transform covariance is not required to be set */
            static constexpr bool COVARIANCE_NOT_REQUIRED = false;

            // -----------------------------------------------------------------------------
            /**
             * \brief Transforms a 3D point covariance using a **certain** transformation (SE(3)).
             * \details
             * If the transformation has no uncertainty, only the **rotation matrix** is used 
             * to rotate the covariance into the new frame.
             *
             * \note **THROW_IF_UNSET** ensures this function is only used with 
             * `se3::Transformation`, which has no uncertainty.
             *
             * \param T_ba The **certain** SE(3) transformation from frame `a` to frame `b`.
             * \param cov_a The **3x3 covariance matrix** in frame `a`.
             * \param p_b The transformed point in frame `b` (unused for certain transformations).
             *
             * \return Transformed **3x3 covariance matrix** in frame `b`.
             */
            template <bool THROW_IF_UNSET = COVARIANCE_REQUIRED>
            CovarianceMatrix transformCovariance(const slam::liemath::se3::Transformation &T_ba,
                                                const CovarianceMatrixConstRef &cov_a,
                                                const HPointConstRef &p_b = HPoint()) {
                (void)&p_b;  // Unused for certain transforms
                static_assert(!THROW_IF_UNSET,
                                "Error: Transformation never has covariance explicitly set.");

                // The covariance is transformed by the rotation matrix
                return T_ba.rotation() * cov_a * T_ba.rotation().transpose();
            }

            // -----------------------------------------------------------------------------
            /**
             * \brief Transforms a 3D point covariance using an **uncertain** transformation.
             * \details
             * If the transformation has **uncertainty**, both the **rotation matrix** and 
             * **transformation uncertainty (covariance)** are used.
             *
             * \note **THROW_IF_UNSET** triggers a runtime error if the transformation's 
             * covariance is not set.
             *
             * \param T_ba The **uncertain** SE(3) transformation from frame `a` to frame `b`.
             * \param cov_a The **3x3 covariance matrix** in frame `a`.
             * \param p_b The transformed point in frame `b`, used for uncertainty propagation.
             *
             * \return Transformed **3x3 covariance matrix** in frame `b`.
             */
            template <bool THROW_IF_UNSET = COVARIANCE_REQUIRED>
            CovarianceMatrix transformCovariance(
                const slam::liemath::se3::TransformationWithCovariance &T_ba,
                const CovarianceMatrixConstRef &cov_a, const HPointConstRef &p_b) {
                // Ensure the transform has covariance, if required
                if (THROW_IF_UNSET && !T_ba.covarianceSet()) {
                    throw std::runtime_error(
                        "Error: TransformationWithCovariance does not have covariance set.");
                }

                // Transform covariance using base transformation (rotation matrix only)
                const auto &T_ba_base = static_cast<const slam::se3::Transformation &>(T_ba);
                CovarianceMatrix cov_b = transformCovariance<false>(T_ba_base, cov_a, p_b);

                // Add uncertainty from the transformation itself
                if (T_ba.covarianceSet()) {
                    auto jacobian = slam::se3::point2fs(p_b.hnormalized()).topRows<3>(); // Compute Jacobian
                    cov_b += jacobian * T_ba.cov() * jacobian.transpose();
                }

                return cov_b;
            }
        }  // namespace r3
    } // namespace liemath
}  // namespace slam
