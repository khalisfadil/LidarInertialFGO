#pragma once

#include <cmath>

#include "Core/Problem/LossFunc/BaseLossFunc.hpp"

namespace slam {
    namespace problem {
        namespace lossfunc {
        
        // -----------------------------------------------------------------------------
        /**
         * \brief Geman-McClure Loss Function for robust estimation.
         * 
         * This function reduces the effect of large residuals while maintaining 
         * quadratic behavior for small errors. It transitions smoothly and is 
         * useful in SLAM, sensor fusion, and robust optimization.
         */
        class GemanMcClureLossFunc : public BaseLossFunc {
            public:

                // -----------------------------------------------------------------------------
                /** \brief Convenience typedefs */
                using Ptr = std::shared_ptr<GemanMcClureLossFunc>;
                using ConstPtr = std::shared_ptr<const GemanMcClureLossFunc>;

                // -----------------------------------------------------------------------------
                /**
                 * \brief Factory method to create a shared instance of GemanMcClureLossFunc.
                 * \param k Robustness threshold (number of standard deviations).
                 * \return A shared pointer to the created GemanMcClureLossFunc instance.
                 */
                static Ptr MakeShared(double k) {
                    return std::make_shared<GemanMcClureLossFunc>(k);
                }

                // -----------------------------------------------------------------------------
                /**
                 * \brief Constructor for Geman-McClure Loss Function.
                 * \param k The threshold parameter controlling robustness.
                 */
                explicit GemanMcClureLossFunc(double k) noexcept : k2_(k * k) {}

                // -----------------------------------------------------------------------------
                /**
                 * \brief Computes the cost function value.
                 * \param whitened_error_norm The whitened error norm.
                 * \return The computed cost.
                 */
                [[nodiscard]] double cost(double whitened_error_norm) const noexcept override {
                    double e2 = whitened_error_norm * whitened_error_norm;
                    return 0.5 * e2 / (k2_ + e2);
                }

                // -----------------------------------------------------------------------------
                /**
                 * \brief Computes the weight for iteratively reweighted least squares (IRLS).
                 * \param whitened_error_norm The whitened error norm.
                 * \return The computed weight.
                 */
                [[nodiscard]] double weight(double whitened_error_norm) const noexcept override {
                    double e2 = whitened_error_norm * whitened_error_norm;
                    double k2e2 = k2_ + e2;
                    return (k2_ * k2_) / (k2e2 * k2e2);
                }

            private:
                // -----------------------------------------------------------------------------
                /** \brief Precomputed squared threshold \( k^2 \) */
                const double k2_;
            };

        }  // namespace lossfunc
    }  // namespace problem
}  // namespace slam
