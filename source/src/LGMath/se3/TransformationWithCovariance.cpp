#include <stdexcept>

#include "LGMath/se3/TransformationWithCovariance.hpp"
#include "LGMath/se3/Operations.hpp"
#include "LGMath/so3/Operations.hpp"

namespace slam {
    namespace liemath {
        namespace se3 {

        ////////////////////////////////////////////////////////////////////////////////
        // Constructors
        ////////////////////////////////////////////////////////////////////////////////

        TransformationWithCovariance::TransformationWithCovariance(bool initCovarianceToZero) noexcept
            : Transformation(),
            covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
            covarianceSet_(initCovarianceToZero) {}

        TransformationWithCovariance::TransformationWithCovariance(const Transformation& T, bool initCovarianceToZero) noexcept
            : Transformation(T),
            covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
            covarianceSet_(initCovarianceToZero) {}

        TransformationWithCovariance::TransformationWithCovariance(Transformation&& T, bool initCovarianceToZero) noexcept
            : Transformation(std::move(T)),
            covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
            covarianceSet_(initCovarianceToZero) {}

        TransformationWithCovariance::TransformationWithCovariance(const Transformation& T,
                                                                const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance) noexcept
            : Transformation(T),
            covariance_(covariance),
            covarianceSet_(true) {}

        TransformationWithCovariance::TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix4d>& T) noexcept
            : Transformation(T),
            covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
            covarianceSet_(false) {}

        TransformationWithCovariance::TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix4d>& T,
                                                                const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance) noexcept
            : Transformation(T),
            covariance_(covariance),
            covarianceSet_(true) {}

        TransformationWithCovariance::TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix3d>& C_ba,
                                                                const Eigen::Ref<const Eigen::Vector3d>& r_ba_ina,
                                                                const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance,
                                                                bool covarianceSet) noexcept
            : Transformation(C_ba, r_ba_ina),
            covariance_(covariance),
            covarianceSet_(covarianceSet) {}

        TransformationWithCovariance::TransformationWithCovariance(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& xi_ab,
                                                                const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance,
                                                                unsigned int numTerms,
                                                                bool covarianceSet) noexcept
            : Transformation(xi_ab, numTerms),
            covariance_(covariance),
            covarianceSet_(covarianceSet) {}

        TransformationWithCovariance::TransformationWithCovariance(const Eigen::Ref<const Eigen::VectorXd>& xi_ab) noexcept
            : Transformation(xi_ab),
            covariance_(Eigen::Matrix<double, 6, 6>::Zero()),
            covarianceSet_(false) {}

        TransformationWithCovariance::TransformationWithCovariance(const Eigen::Ref<const Eigen::VectorXd>& xi_ab,
                                                                const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance) noexcept
            : Transformation(xi_ab),
            covariance_(covariance),
            covarianceSet_(true) {}

        ////////////////////////////////////////////////////////////////////////////////
        // Assignment Operators
        ////////////////////////////////////////////////////////////////////////////////

        TransformationWithCovariance& TransformationWithCovariance::operator=(const Transformation& T) noexcept {
            Transformation::operator=(T);
            covariance_.setZero();
            covarianceSet_ = false;
            return *this;
        }

        TransformationWithCovariance& TransformationWithCovariance::operator=(Transformation&& T) noexcept {
            Transformation::operator=(std::move(T));
            covariance_.setZero();
            covarianceSet_ = false;
            return *this;
        }

        ////////////////////////////////////////////////////////////////////////////////
        // Covariance Management
        ////////////////////////////////////////////////////////////////////////////////

        const Eigen::Matrix<double, 6, 6>& TransformationWithCovariance::cov() const noexcept {
            return covariance_;
        }

        bool TransformationWithCovariance::covarianceSet() const noexcept {
            return covarianceSet_;
        }

        void TransformationWithCovariance::setCovariance(const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& covariance) noexcept {
            covariance_ = covariance;
            covarianceSet_ = true;
        }

        void TransformationWithCovariance::setZeroCovariance() noexcept {
            covariance_.setZero();
            covarianceSet_ = true;
        }

        ////////////////////////////////////////////////////////////////////////////////
        // Operations
        ////////////////////////////////////////////////////////////////////////////////

        TransformationWithCovariance TransformationWithCovariance::inverse() const {
            TransformationWithCovariance temp(Transformation::inverse(), false);
            if (covarianceSet_) {
                Eigen::Matrix<double, 6, 6> adjointOfInverse = temp.adjoint();
                temp.setCovariance(adjointOfInverse * covariance_ * adjointOfInverse.transpose());
            }
            return temp;
        }

        TransformationWithCovariance& TransformationWithCovariance::operator*=(const TransformationWithCovariance& T_rhs) {
            if (covarianceSet_ || T_rhs.covarianceSet_) {
                Eigen::Matrix<double, 6, 6> Ad_lhs = Transformation::adjoint();
                covariance_ = Ad_lhs * (covariance_ + T_rhs.covariance_) * Ad_lhs.transpose();
                covarianceSet_ = true;
            }
            Transformation::operator*=(T_rhs);
            return *this;
        }

        TransformationWithCovariance& TransformationWithCovariance::operator*=(const Transformation& T_rhs) noexcept {
            Transformation::operator*=(T_rhs);
            return *this;
        }

        TransformationWithCovariance& TransformationWithCovariance::operator/=(const TransformationWithCovariance& T_rhs) {
            if (covarianceSet_ || T_rhs.covarianceSet_) {
                Transformation T_inv = T_rhs.inverse();
                Eigen::Matrix<double, 6, 6> Ad_lhs_rhs = T_inv.adjoint();
                covariance_ = Ad_lhs_rhs * (covariance_ + T_rhs.covariance_) * Ad_lhs_rhs.transpose();
                covarianceSet_ = true;
            }
            Transformation::operator/=(T_rhs);
            return *this;
        }

        TransformationWithCovariance& TransformationWithCovariance::operator/=(const Transformation& T_rhs) noexcept {
            Transformation::operator/=(T_rhs);
            return *this;
        }

        ////////////////////////////////////////////////////////////////////////////////
        // Standalone Operators
        ////////////////////////////////////////////////////////////////////////////////

        TransformationWithCovariance operator*(const TransformationWithCovariance& T_lhs, const TransformationWithCovariance& T_rhs) {
            TransformationWithCovariance result = T_lhs;
            result *= T_rhs;
            return result;
        }

        TransformationWithCovariance operator*(const TransformationWithCovariance& T_lhs, const Transformation& T_rhs) noexcept {
            TransformationWithCovariance result = T_lhs;
            result *= T_rhs;
            return result;
        }

        TransformationWithCovariance operator*(const Transformation& T_lhs, const TransformationWithCovariance& T_rhs) {
            TransformationWithCovariance result(T_lhs, false);
            result *= T_rhs;
            return result;
        }

        TransformationWithCovariance operator/(const TransformationWithCovariance& T_lhs, const TransformationWithCovariance& T_rhs) {
            TransformationWithCovariance result = T_lhs;
            result /= T_rhs;
            return result;
        }

        TransformationWithCovariance operator/(const TransformationWithCovariance& T_lhs, const Transformation& T_rhs) noexcept {
            TransformationWithCovariance result = T_lhs;
            result /= T_rhs;
            return result;
        }

        TransformationWithCovariance operator/(const Transformation& T_lhs, const TransformationWithCovariance& T_rhs) {
            TransformationWithCovariance result(T_lhs, false);
            result /= T_rhs;
            return result;
        }

        }  // namespace se3
    }  // namespace liemath
}  // namespace slam

std::ostream& operator<<(std::ostream& out, const slam::liemath::se3::TransformationWithCovariance& T) {
    out << "\n" << T.matrix();
    if (T.covarianceSet()) {
        out << "\n" << T.cov();
    } else {
        out << "\nCovariance is unset.";
    }
    return out;
}