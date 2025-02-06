#include <cmath>

#include "source/include/LGMath/CommonMath.hpp"

namespace slam {
    namespace liemath {
        namespace common {

            // ----------------------------------------------------------------------------
            // angleMod
            // ----------------------------------------------------------------------------

            double angleMod(double radians) {
                return radians - (constants::TWO_PI * std::round(radians * constants::ONE_DIV_TWO_PI));
            }

            // ----------------------------------------------------------------------------
            // deg2rad
            // ----------------------------------------------------------------------------

            double deg2rad(double degrees) {
                return degrees * constants::DEG2RAD;
            }

            // ----------------------------------------------------------------------------
            // rad2deg
            // ----------------------------------------------------------------------------

            double rad2deg(double radians) {
                return radians * constants::RAD2DEG;
            }

            // ----------------------------------------------------------------------------
            // nearEqual
            // ----------------------------------------------------------------------------

            bool nearEqual(double a, double b, double tol) {
                return std::fabs(a - b) <= tol;
            }

            // ----------------------------------------------------------------------------
            // nearEqual
            // ----------------------------------------------------------------------------

            bool nearEqual(const Eigen::Ref<const Eigen::MatrixXd>& A, 
                            const Eigen::Ref<const Eigen::MatrixXd>& B, 
                            double tol = 1e-6) {
                return A.rows() == B.rows() && A.cols() == B.cols() && A.isApprox(B, tol);
            }

            // ----------------------------------------------------------------------------
            // nearEqualAngle
            // ----------------------------------------------------------------------------

            bool nearEqualAngle(double radA, double radB, double tol) {
                return nearEqual(angleMod(radA - radB), 0.0, tol);
            }

            // ----------------------------------------------------------------------------
            // nearEqualAxisAngle
            // ----------------------------------------------------------------------------

            bool nearEqualAxisAngle(const Eigen::Ref<const Eigen::Vector3d>& aaxis1,
                                    const Eigen::Ref<const Eigen::Vector3d>& aaxis2,
                                    double tol) {
                constexpr double EPSILON = 1e-12;

                double a1 = aaxis1.norm();
                double a2 = aaxis2.norm();

                // If both angles are near zero, return true
                if (std::fabs(a1) < EPSILON && std::fabs(a2) < EPSILON) {
                    return true;
                }

                // Normalize axis and compare
                Eigen::Vector3d axis1 = aaxis1 / a1;
                Eigen::Vector3d axis2 = aaxis2 / a2;

                return axis1.isApprox(axis2, tol) && nearEqualAngle(a1, a2, tol);
            }

            // ----------------------------------------------------------------------------
            // nearEqualLieAlg
            // ----------------------------------------------------------------------------

            bool nearEqualLieAlg(const Eigen::Matrix<double, 6, 1>& vec1,
                                const Eigen::Matrix<double, 6, 1>& vec2, double tol) {
                return nearEqualAxisAngle(vec1.tail<3>(), vec2.tail<3>(), tol) &&
                    nearEqual(vec1.head<3>(), vec2.head<3>(), tol);
            }

        }  // namespace common
    } // namespace liemath
}  // namespace slam
