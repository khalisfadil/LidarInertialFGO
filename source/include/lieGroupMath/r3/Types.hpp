
#pragma once

#include <Eigen/Dense>

namespace slam {
    namespace r3 {

        // -----------------------------------------------------------------------------
        /// **📌 3D Point Representations**
        /**
         * \brief Represents a **3D point** in Euclidean space **ℝ³**.
         */
        using Point = Eigen::Vector3d;

        // -----------------------------------------------------------------------------
        /**
         * \brief Mutable reference to a **3D point**.
         */
        using PointRef = Eigen::Ref<Point>;

        // -----------------------------------------------------------------------------
        /**
         * \brief Immutable reference to a **3D point**.
         */
        using PointConstRef = Eigen::Ref<const Point>;

        // -----------------------------------------------------------------------------
        /// **📌 3D Homogeneous Point Representations**
        /**
         * \brief Represents a **3D homogeneous point** (4D vector).
         * \details Typically used in projective geometry or SE(3) transformations.
         */
        using HPoint = Eigen::Vector4d;

        // -----------------------------------------------------------------------------
        /**
         * \brief Mutable reference to a **homogeneous point**.
         */
        using HPointRef = Eigen::Ref<HPoint>;

        // -----------------------------------------------------------------------------
        /**
         * \brief Immutable reference to a **homogeneous point**.
         */
        using HPointConstRef = Eigen::Ref<const HPoint>;

        // -----------------------------------------------------------------------------
        /// **📌 3D Point Covariance Matrix Representations**
        /**
         * \brief Represents a **3x3 covariance matrix** of a **3D point**.
         */
        using CovarianceMatrix = Eigen::Matrix3d;

        // -----------------------------------------------------------------------------
        /**
         * \brief Mutable reference to a **3D covariance matrix**.
         */
        using CovarianceMatrixRef = Eigen::Ref<CovarianceMatrix>;

        // -----------------------------------------------------------------------------
        /**
         * \brief Immutable reference to a **3D covariance matrix**.
         */
        using CovarianceMatrixConstRef = Eigen::Ref<const CovarianceMatrix>;

    }  // namespace r3
}  // namespace slam
