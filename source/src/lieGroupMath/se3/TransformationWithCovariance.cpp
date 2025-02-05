#include <include/lieGroupMath/se3/TransformationWithCovariance.hpp>

#include <stdexcept>

#include <include/lieGroupMath/se3/Operations.hpp>
#include <include/lieGroupMath/so3/Operations.hpp>

namespace slam {
    namespace se3 {

        // ----------------------------------------------------------------------------
        // Default Constructor
        // ----------------------------------------------------------------------------

        TransformationWithCovariance::TransformationWithCovariance(bool initCovarianceToZero)
            : Transformation(),
            covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
            covarianceSet_(initCovarianceToZero) {}

        // ----------------------------------------------------------------------------
        // Constructors from Transformation
        // ----------------------------------------------------------------------------

        TransformationWithCovariance::TransformationWithCovariance(const Transformation& T, bool initCovarianceToZero)
            : Transformation(T),
            covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
            covarianceSet_(initCovarianceToZero) {}

        // ----------------------------------------------------------------------------

        TransformationWithCovariance::TransformationWithCovariance(Transformation&& T, bool initCovarianceToZero)
            : Transformation(std::move(T)),
            covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
            covarianceSet_(initCovarianceToZero) {}

        // ----------------------------------------------------------------------------

        TransformationWithCovariance::TransformationWithCovariance(const Transformation& T, const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance)
            : Transformation(T), covariance_(covariance), covarianceSet_(true) {}

        // ----------------------------------------------------------------------------
        // Constructors from Matrix Representations
        // ----------------------------------------------------------------------------

        TransformationWithCovariance::TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix4d>& T)
            : Transformation(T),
            covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
            covarianceSet_(false) {}

        // ----------------------------------------------------------------------------

        TransformationWithCovariance::TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix4d>& T,
                                                                const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance)
            : Transformation(T), covariance_(covariance), covarianceSet_(true) {}

        // ----------------------------------------------------------------------------
        // Constructors from Rotation and Translation
        // ----------------------------------------------------------------------------

        TransformationWithCovariance::TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix3d>& C_ba,
                                                                const Eigen::Ref<const Eigen::Vector3d>& r_ba_ina)
            : Transformation(C_ba, r_ba_ina),
            covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
            covarianceSet_(false) {}

        // ----------------------------------------------------------------------------

        TransformationWithCovariance::TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix3d>& C_ba,
                                                                const Eigen::Ref<const Eigen::Vector3d>& r_ba_ina,
                                                                const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance)
            : Transformation(C_ba, r_ba_ina),
            covariance_(covariance),
            covarianceSet_(true) {}

        // ----------------------------------------------------------------------------
        // Constructors from Lie Algebra Vector
        // ----------------------------------------------------------------------------

        TransformationWithCovariance::TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ab,
                                                                unsigned int numTerms)
            : Transformation(xi_ab, numTerms),
            covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
            covarianceSet_(false) {}

        // ----------------------------------------------------------------------------

        TransformationWithCovariance::TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ab,
                                                                const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance,
                                                                unsigned int numTerms)
            : Transformation(xi_ab, numTerms),
            covariance_(covariance),
            covarianceSet_(true) {}

        // ----------------------------------------------------------------------------

        TransformationWithCovariance::TransformationWithCovariance(const Eigen::Ref<const Eigen::VectorXd>& xi_ab)
            : Transformation(xi_ab),
            covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
            covarianceSet_(false) {}

        // ----------------------------------------------------------------------------

        TransformationWithCovariance::TransformationWithCovariance(const Eigen::Ref<const Eigen::VectorXd>& xi_ab,
                                                                const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance)
            : Transformation(xi_ab),
            covariance_(covariance),
            covarianceSet_(true) {}

        // ----------------------------------------------------------------------------
        // Assignment Operators
        // ----------------------------------------------------------------------------

        TransformationWithCovariance& TransformationWithCovariance::operator=(const Transformation& T) noexcept {
            Transformation::operator=(T);
            covariance_.setZero();
            covarianceSet_ = false;
            return *this;
        }

        // ----------------------------------------------------------------------------

        TransformationWithCovariance& TransformationWithCovariance::operator=(Transformation&& T) noexcept {
            Transformation::operator=(std::move(T));
            covariance_.setZero();
            covarianceSet_ = false;
            return *this;
        }

        // ----------------------------------------------------------------------------
        // Covariance Management
        // ----------------------------------------------------------------------------

        const Eigen::Matrix<double, 6, 6>& TransformationWithCovariance::cov() const {
            if (!covarianceSet_) {
                throw std::logic_error("Covariance accessed before being set. Use setCovariance or initialize with a covariance.");
            }
            return covariance_;
        }

        // ----------------------------------------------------------------------------

        bool TransformationWithCovariance::covarianceSet() const noexcept {
            return covarianceSet_;
        }

        // ----------------------------------------------------------------------------

        void TransformationWithCovariance::setCovariance(const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance) {
            covariance_ = covariance;
            covarianceSet_ = true;
        }
        
        // ----------------------------------------------------------------------------

        void TransformationWithCovariance::setZeroCovariance() noexcept {
            covariance_.setZero();
            covarianceSet_ = true;
        }

        // ----------------------------------------------------------------------------
        // Inverse Transformation
        // ----------------------------------------------------------------------------

        TransformationWithCovariance TransformationWithCovariance::inverse() const {
            TransformationWithCovariance temp(Transformation::inverse(), false);
            Eigen::Matrix<double, 6, 6> adjointOfInverse = temp.adjoint();
            temp.setCovariance(adjointOfInverse * covariance_ * adjointOfInverse.transpose());
            return temp;
        }

        // ----------------------------------------------------------------------------
        // Multiplication Operators
        // ----------------------------------------------------------------------------
        TransformationWithCovariance& TransformationWithCovariance::operator*=(const TransformationWithCovariance& T_rhs) {
            Eigen::Matrix<double, 6, 6> Ad_lhs = Transformation::adjoint();
            covariance_ += Ad_lhs * T_rhs.covariance_ * Ad_lhs.transpose();
            covarianceSet_ = covarianceSet_ && T_rhs.covarianceSet_;
            Transformation::operator*=(T_rhs);
            return *this;
        }

        // ----------------------------------------------------------------------------
        
        TransformationWithCovariance& TransformationWithCovariance::operator*=(const Transformation& T_rhs) {
            Transformation::operator*=(T_rhs);
            return *this;
        }

        // ----------------------------------------------------------------------------
        // Division Operators
        // ----------------------------------------------------------------------------

        TransformationWithCovariance& TransformationWithCovariance::operator/=(const TransformationWithCovariance& T_rhs) {
            Transformation::operator/=(T_rhs);
            Eigen::Matrix<double, 6, 6> Ad_lhs_rhs = Transformation::adjoint();
            covariance_ += Ad_lhs_rhs * T_rhs.covariance_ * Ad_lhs_rhs.transpose();
            covarianceSet_ = covarianceSet_ && T_rhs.covarianceSet_;
            return *this;
        }

        // ----------------------------------------------------------------------------

        TransformationWithCovariance& TransformationWithCovariance::operator/=(const Transformation& T_rhs) {
            Transformation::operator/=(T_rhs);
            return *this;
        }

        // ----------------------------------------------------------------------------
        // Standalone Operators
        // ----------------------------------------------------------------------------

        TransformationWithCovariance operator*(TransformationWithCovariance T_lhs, const TransformationWithCovariance& T_rhs) {
            T_lhs *= T_rhs;
            return T_lhs;
        }

        // ----------------------------------------------------------------------------

        TransformationWithCovariance operator/(TransformationWithCovariance T_lhs, const TransformationWithCovariance& T_rhs) {
            T_lhs /= T_rhs;
            return T_lhs;
        }

    }  // namespace se3
}  // namespace slam

std::ostream& operator<<(std::ostream& out, const slam::se3::TransformationWithCovariance& T) {
    out << "\n" << T.matrix();
    if (T.covarianceSet()) {
        out << "\n" << T.cov();
    } else {
        out << "\nCovariance is unset.";
    }
    return out;
}

