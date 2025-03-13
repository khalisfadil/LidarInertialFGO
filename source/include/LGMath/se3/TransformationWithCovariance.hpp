#pragma once

#include <Eigen/Core>
#include "LGMath/se3/Transformation.hpp"

namespace slam {
    namespace liemath {
        namespace se3 {

        // ----------------------------------------------------------------------------
        /**
         * \class TransformationWithCovariance
         * \brief A transformation matrix class with associated covariance.
         * \details Extends `Transformation` to include covariance propagation.
         * \note This class introduces additional matrix operations, making it slower than `Transformation`.
         */
        class TransformationWithCovariance : public Transformation {
        public:
            // ----------------------------------------------------------------------------
            /** \brief Default constructor (optionally initializes covariance to zero) */
            TransformationWithCovariance(bool initCovarianceToZero = false) noexcept;

            // ----------------------------------------------------------------------------
            /** \brief Copy constructor */
            TransformationWithCovariance(const TransformationWithCovariance&) = default;

            // ----------------------------------------------------------------------------
            /** \brief Move constructor */
            TransformationWithCovariance(TransformationWithCovariance&&) noexcept = default;

            // ----------------------------------------------------------------------------
            /** \brief Copy constructor from basic Transformation */
            TransformationWithCovariance(const Transformation& T, bool initCovarianceToZero = false) noexcept;

            // ----------------------------------------------------------------------------
            /** \brief Move constructor from basic Transformation */
            TransformationWithCovariance(Transformation&& T, bool initCovarianceToZero = false) noexcept;

            // ----------------------------------------------------------------------------
            /** \brief Copy constructor from basic Transformation with covariance */
            TransformationWithCovariance(const Transformation& T, const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance) noexcept;

            // ----------------------------------------------------------------------------
            /** \brief Constructor from a 4x4 transformation matrix */
            explicit TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix4d>& T) noexcept;

            // ----------------------------------------------------------------------------
            /** \brief Constructor from a 4x4 transformation matrix with covariance */
            TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix4d>& T,
                                        const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance) noexcept;

            // ----------------------------------------------------------------------------
            /**
             * \brief Constructor from rotation and translation with optional covariance
             * \details The transformation is initialized as:
             * \f$ T_{ba} = \begin{bmatrix} C_{ba} & -C_{ba} r_{ba}^{a} \\ 0 & 1 \end{bmatrix} \f$
             */
            TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix3d>& C_ba,
                                        const Eigen::Ref<const Eigen::Vector3d>& r_ba_ina,
                                        const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance = Eigen::Matrix<double, 6, 6>::Zero(),
                                        bool covarianceSet = false) noexcept;

            // ----------------------------------------------------------------------------
            /**
             * \brief Constructor from a Lie algebra vector with optional covariance
             * \details The transformation is computed using `vec2tran(xi_ab)`.
             */
            TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ab,
                                        const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance = Eigen::Matrix<double, 6, 6>::Zero(),
                                        unsigned int numTerms = 0,
                                        bool covarianceSet = false) noexcept;

            // ----------------------------------------------------------------------------
            /** \brief Constructor from a general vector (must be of size 6) */
            explicit TransformationWithCovariance(const Eigen::Ref<const Eigen::VectorXd>& xi_ab) noexcept;

            // ----------------------------------------------------------------------------
            /** \brief Constructor from a general vector (must be of size 6) with covariance */
            TransformationWithCovariance(const Eigen::Ref<const Eigen::VectorXd>& xi_ab,
                                        const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance) noexcept;

            // ----------------------------------------------------------------------------
            /** \brief Destructor */
            ~TransformationWithCovariance() noexcept = default;

            // ----------------------------------------------------------------------------
            /** \brief Copy assignment operator */
            TransformationWithCovariance& operator=(const TransformationWithCovariance&) = default;

            // ----------------------------------------------------------------------------
            /** \brief Move assignment operator */
            TransformationWithCovariance& operator=(TransformationWithCovariance&&) noexcept = default;

            // ----------------------------------------------------------------------------
            /**
             * \brief Copy assignment operator from `Transformation`
             * \details Resets covariance to zero and unsets it.
             */
            TransformationWithCovariance& operator=(const Transformation& T) noexcept override;

            // ----------------------------------------------------------------------------
            /**
             * \brief Move assignment operator from `Transformation`
             * \details Resets covariance to zero and unsets it.
             */
            TransformationWithCovariance& operator=(Transformation&& T) noexcept override;

            // ----------------------------------------------------------------------------
            /** \brief Get the covariance matrix */
            const Eigen::Matrix<double, 6, 6>& cov() const noexcept;

            // ----------------------------------------------------------------------------
            /** \brief Check if covariance has been set */
            bool covarianceSet() const noexcept;

            // ----------------------------------------------------------------------------
            /** \brief Set the covariance matrix */
            void setCovariance(const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance) noexcept;

            // ----------------------------------------------------------------------------
            /** \brief Set covariance to zero (perfect certainty) */
            void setZeroCovariance() noexcept;

            // ----------------------------------------------------------------------------
            /** \brief Compute the inverse transformation with covariance */
            TransformationWithCovariance inverse() const;

            // ----------------------------------------------------------------------------
            /** \brief In-place right-hand side multiplication with another `TransformationWithCovariance` */
            TransformationWithCovariance& operator*=(const TransformationWithCovariance& T_rhs);

            // ----------------------------------------------------------------------------
            /**
             * \brief In-place right-hand side multiplication with a basic `Transformation`
             * \note Assumes `T_rhs` has zero covariance (perfect certainty).
             */
            TransformationWithCovariance& operator*=(const Transformation& T_rhs) noexcept override;

            // ----------------------------------------------------------------------------
            /** \brief In-place right-hand side division by another `TransformationWithCovariance` */
            TransformationWithCovariance& operator/=(const TransformationWithCovariance& T_rhs);

            // ----------------------------------------------------------------------------
            /**
             * \brief In-place right-hand side division by a basic `Transformation`
             * \note Assumes `T_rhs` has zero covariance (perfect certainty).
             */
            TransformationWithCovariance& operator/=(const Transformation& T_rhs) noexcept override;

        private:
            // ----------------------------------------------------------------------------
            /** \brief Covariance matrix (6x6) */
            Eigen::Matrix<double, 6, 6> covariance_;

            // ----------------------------------------------------------------------------
            /** \brief Flag indicating whether covariance has been set */
            bool covarianceSet_ = false;
        };

        // ----------------------------------------------------------------------------
        // Standalone Operator Overloads
        // ----------------------------------------------------------------------------

        /** \brief Multiply two `TransformationWithCovariance` objects */
        TransformationWithCovariance operator*(const TransformationWithCovariance& T_lhs, const TransformationWithCovariance& T_rhs);

        /** \brief Multiply `TransformationWithCovariance` by a basic `Transformation` */
        TransformationWithCovariance operator*(const TransformationWithCovariance& T_lhs, const Transformation& T_rhs) noexcept;

        /** \brief Multiply `Transformation` by a `TransformationWithCovariance` */
        TransformationWithCovariance operator*(const Transformation& T_lhs, const TransformationWithCovariance& T_rhs);

        /** \brief Divide `TransformationWithCovariance` by another `TransformationWithCovariance` */
        TransformationWithCovariance operator/(const TransformationWithCovariance& T_lhs, const TransformationWithCovariance& T_rhs);

        /** \brief Divide `TransformationWithCovariance` by a basic `Transformation` */
        TransformationWithCovariance operator/(const TransformationWithCovariance& T_lhs, const Transformation& T_rhs) noexcept;

        /** \brief Divide `Transformation` by a `TransformationWithCovariance` */
        TransformationWithCovariance operator/(const Transformation& T_lhs, const TransformationWithCovariance& T_rhs);

        /** \brief Print transformation and covariance */
        std::ostream& operator<<(std::ostream& out, const TransformationWithCovariance& T);

        }  // namespace se3
    }  // namespace liemath
}  // namespace slam