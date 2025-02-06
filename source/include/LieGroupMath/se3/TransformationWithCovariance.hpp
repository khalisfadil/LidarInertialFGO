#pragma once

#include <Eigen/Core>

#include <include/lieGroupMath/se3/Transformation.hpp>

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
                    TransformationWithCovariance(bool initCovarianceToZero = false);

                    // ----------------------------------------------------------------------------
                    /** \brief Copy constructor */
                    TransformationWithCovariance(const TransformationWithCovariance&) = default;

                    // ----------------------------------------------------------------------------
                    /** \brief Move constructor */
                    TransformationWithCovariance(TransformationWithCovariance&&) = default;

                    // ----------------------------------------------------------------------------
                    /** \brief Copy constructor from basic Transformation */
                    TransformationWithCovariance(const Transformation& T, bool initCovarianceToZero = false);

                    // ----------------------------------------------------------------------------
                    /** \brief Move constructor from basic Transformation */
                    TransformationWithCovariance(Transformation&& T, bool initCovarianceToZero = false);

                    // ----------------------------------------------------------------------------
                    /** \brief Copy constructor from basic Transformation, with covariance */
                    TransformationWithCovariance(const Transformation& T, const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance);

                    // ----------------------------------------------------------------------------
                    /** \brief Constructor from a 4x4 transformation matrix */
                    explicit TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix4d>& T);

                    // ----------------------------------------------------------------------------
                    /** \brief Constructor from a 4x4 transformation matrix with covariance */
                    TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix4d>& T, const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance);

                    // ----------------------------------------------------------------------------
                    /**
                     * \brief Constructor from rotation and translation
                     * \details The transformation is initialized as:
                     * \f$ T_{ba} = \begin{bmatrix} C_{ba} & -C_{ba} r_{ba}^{a} \\ 0 & 1 \end{bmatrix} \f$
                     */
                    TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix3d>& C_ba, const Eigen::Ref<const Eigen::Vector3d>& r_ba_ina);

                    // ----------------------------------------------------------------------------
                    /**
                     * \brief Constructor from rotation and translation, with covariance
                     */
                    TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix3d>& C_ba, const Eigen::Ref<const Eigen::Vector3d>& r_ba_ina,
                                                const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance);

                    // ----------------------------------------------------------------------------
                    /**
                     * \brief Constructor from a Lie algebra vector
                     * \details The transformation is computed using `vec2tran(xi_ab)`.
                     */
                    TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ab, unsigned int numTerms = 0);

                    // ----------------------------------------------------------------------------
                    /**
                     * \brief Constructor from a Lie algebra vector with covariance
                     */
                    TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ab, const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance,
                                                unsigned int numTerms = 0);

                    // ----------------------------------------------------------------------------
                    /**
                     * \brief Constructor from a general vector (must be of size 6)
                     */
                    explicit TransformationWithCovariance(const Eigen::Ref<const Eigen::VectorXd>& xi_ab);

                    // ----------------------------------------------------------------------------
                    /**
                     * \brief Constructor from a general vector (must be of size 6) with covariance
                     */
                    TransformationWithCovariance(const Eigen::Ref<const Eigen::VectorXd>& xi_ab, const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance);

                    // ----------------------------------------------------------------------------
                    /** \brief Destructor */
                    ~TransformationWithCovariance() override = default;

                    // ----------------------------------------------------------------------------
                    /** \brief Copy assignment operator */
                    TransformationWithCovariance& operator=(const TransformationWithCovariance&) = default;

                    // ----------------------------------------------------------------------------
                    /** \brief Move assignment operator */
                    TransformationWithCovariance& operator=(TransformationWithCovariance&&) = default;

                    // ----------------------------------------------------------------------------
                    /**
                     * \brief Copy assignment operator from `Transformation`
                     * \details This resets the covariance, requiring explicit re-initialization before querying it.
                     */
                    TransformationWithCovariance& operator=(const Transformation& T) noexcept override;

                    // ----------------------------------------------------------------------------
                    /**
                     * \brief Move assignment operator from `Transformation`
                     * \details This resets the covariance, requiring explicit re-initialization before querying it.
                     */
                    TransformationWithCovariance& operator=(Transformation&& T) noexcept override;

                    // ----------------------------------------------------------------------------
                    /** \brief Get the covariance matrix */
                    const Eigen::Matrix<double, 6, 6>& cov() const;

                    // ----------------------------------------------------------------------------
                    /** \brief Check if covariance has been set */
                    bool covarianceSet() const noexcept;

                    // ----------------------------------------------------------------------------
                    /** \brief Set the covariance matrix */
                    void setCovariance(const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance);

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
                     * \note Assumes that `T_rhs` has perfect certainty (zero covariance).
                     */
                    TransformationWithCovariance& operator*=(const Transformation& T_rhs) noexcept override;

                    // ----------------------------------------------------------------------------
                    /** \brief In-place right-hand side multiplication by the inverse of another `TransformationWithCovariance` */
                    TransformationWithCovariance& operator/=(const TransformationWithCovariance& T_rhs);

                    // ----------------------------------------------------------------------------
                    /**
                     * \brief In-place right-hand side multiplication by the inverse of a basic `Transformation`
                     * \note Assumes that `T_rhs` has perfect certainty (zero covariance).
                     */
                    TransformationWithCovariance& operator/=(const Transformation& T_rhs) noexcept override;

                private:

                    // ----------------------------------------------------------------------------
                    /** \brief Covariance matrix (6x6) */
                    Eigen::Matrix<double, 6, 6> covariance_;

                    // ----------------------------------------------------------------------------
                    /** \brief Flag indicating whether covariance has been set */
                    bool covarianceSet_;
            };

            // ----------------------------------------------------------------------------
            // Standalone Operator Overloads
            // ----------------------------------------------------------------------------

            // ----------------------------------------------------------------------------
            /** \brief Multiply two `TransformationWithCovariance` objects */
            TransformationWithCovariance operator*(TransformationWithCovariance T_lhs, const TransformationWithCovariance& T_rhs);

            // ----------------------------------------------------------------------------
            /** \brief Multiply `TransformationWithCovariance` by a basic `Transformation` */
            TransformationWithCovariance operator*(TransformationWithCovariance T_lhs, const Transformation& T_rhs);

            // ----------------------------------------------------------------------------
            /** \brief Multiply `Transformation` by a `TransformationWithCovariance` */
            TransformationWithCovariance operator*(const Transformation& T_lhs, const TransformationWithCovariance& T_rhs);

            // ----------------------------------------------------------------------------
            /** \brief Divide `TransformationWithCovariance` by another `TransformationWithCovariance` */
            TransformationWithCovariance operator/(TransformationWithCovariance T_lhs, const TransformationWithCovariance& T_rhs);
            
            // ----------------------------------------------------------------------------
            /** \brief Divide `TransformationWithCovariance` by a basic `Transformation` */
            TransformationWithCovariance operator/(TransformationWithCovariance T_lhs, const Transformation& T_rhs);

            // ----------------------------------------------------------------------------
            /** \brief Divide `Transformation` by a `TransformationWithCovariance` */
            TransformationWithCovariance operator/(const Transformation& T_lhs, const TransformationWithCovariance& T_rhs);

            // ----------------------------------------------------------------------------
            /** \brief Print transformation and covariance */
            std::ostream& operator<<(std::ostream& out, const slam::liemath::se3::TransformationWithCovariance& T);

        }  // namespace se3
    } // liemath
}  // namespace slam

