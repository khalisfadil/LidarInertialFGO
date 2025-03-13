#include "LGMath/CommonMath.hpp"
#include <cmath>

namespace slam {
    namespace liemath {
        namespace common {

            // ----------------------------------------------------------------------------
            // Angle Wrapping
            // ----------------------------------------------------------------------------

            double angleMod(double radians) noexcept {
                return radians - constants::TWO_PI * std::round(radians * constants::ONE_DIV_TWO_PI);
            }

            // ----------------------------------------------------------------------------
            // Degree-Radian Conversions
            // ----------------------------------------------------------------------------

            double deg2rad(double degrees) noexcept {
                return degrees * constants::DEG2RAD;
            }

            double rad2deg(double radians) noexcept {
                return radians * constants::RAD2DEG;
            }

            // ----------------------------------------------------------------------------
            // Near Equality Comparisons
            // ----------------------------------------------------------------------------

            bool nearEqual(double a, double b, double tol) noexcept {
                return std::fabs(a - b) <= tol;
            }

            bool nearEqual(const Eigen::Ref<const Eigen::MatrixXd>& A,
                        const Eigen::Ref<const Eigen::MatrixXd>& B,
                        double tol) noexcept {
                return A.rows() == B.rows() && A.cols() == B.cols() && A.isApprox(B, tol);
            }

            bool nearEqualAngle(double radA, double radB, double tol) noexcept {
                return nearEqual(angleMod(radA - radB), 0.0, tol);
            }

            bool nearEqualAxisAngle(const Eigen::Ref<const Eigen::Vector3d>& aaxis1,
                                    const Eigen::Ref<const Eigen::Vector3d>& aaxis2,
                                    double tol) noexcept {
                constexpr double EPSILON = 1e-12;

                double a1 = aaxis1.norm();
                double a2 = aaxis2.norm();

                if (a1 < EPSILON && a2 < EPSILON) {
                    return true;  // Both angles near zero
                }

                // Normalize axes and compare angle magnitudes
                Eigen::Vector3d axis1 = aaxis1 / a1;
                Eigen::Vector3d axis2 = aaxis2 / a2;
                return axis1.isApprox(axis2, tol) && nearEqual(a1, a2, tol);
            }

            bool nearEqualLieAlg(const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& vec1,
                                const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& vec2,
                                double tol) noexcept {
                return nearEqualAxisAngle(vec1.tail<3>(), vec2.tail<3>(), tol) &&
                    nearEqual(vec1.head<3>(), vec2.head<3>(), tol);
            }

        }  // namespace common
    }  // namespace liemath
}  // namespace slam