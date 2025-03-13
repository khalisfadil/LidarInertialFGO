#pragma once

#include <cmath>

#include "Problem/LossFunc/BaseLossFunc.hpp"

namespace slam {
    namespace problem {
        namespace lossfunc {
        
        // -----------------------------------------------------------------------------
        /**
         * \brief Dynamic Covariance Scaling (DCS) Loss Function.
         * 
         * The DCS loss function is designed for robust estimation, reducing 
         * the effect of spurious measurements while maintaining efficiency.
         * It transitions from quadratic loss for small residuals to a 
         * smoothly bounded loss for large residuals.
         */
        class DcsLossFunc : public BaseLossFunc {
            public:
                // -----------------------------------------------------------------------------
                /** \brief Convenience typedefs */
                using Ptr = std::shared_ptr<DcsLossFunc>;
                using ConstPtr = std::shared_ptr<const DcsLossFunc>;

                // -----------------------------------------------------------------------------
                /**
                 * \brief Factory method to create a shared instance of DcsLossFunc.
                 * \param k Robustness threshold (number of standard deviations).
                 * \return A shared pointer to the created DcsLossFunc instance.
                 */
                static Ptr MakeShared(double k) {
                    return std::make_shared<DcsLossFunc>(k);
                }

                // -----------------------------------------------------------------------------
                /**
                 * \brief Constructor for DCS Loss Function.
                 * \param k The threshold parameter controlling the robustness.
                 */
                explicit DcsLossFunc(double k) noexcept : k2_(k * k) {}

                // -----------------------------------------------------------------------------
                /**
                 * \brief Computes the cost function value.
                 * \param whitened_error_norm The whitened error norm.
                 * \return The computed cost.
                 */
                [[nodiscard]] double cost(double whitened_error_norm) const noexcept override {
                    double e2 = whitened_error_norm * whitened_error_norm;
                    return (e2 <= k2_)
                            ? 0.5 * e2
                            : 2.0 * k2_ * e2 / (k2_ + e2) - 0.5 * k2_;
                }

                // -----------------------------------------------------------------------------
                /**
                 * \brief Computes the weight for iteratively reweighted least squares (IRLS).
                 * \param whitened_error_norm The whitened error norm.
                 * \return The computed weight.
                 */
                [[nodiscard]] double weight(double whitened_error_norm) const noexcept override {
                    double e2 = whitened_error_norm * whitened_error_norm;
                    if (e2 <= k2_) {
                    return 1.0;
                    } else {
                    double k2e2 = k2_ + e2;
                    return 4.0 * k2_ * k2_ / (k2e2 * k2e2);
                    }
                }

            private:
            
                // -----------------------------------------------------------------------------
                /** \brief Precomputed squared threshold \( k^2 \) */
                const double k2_;
            };

        }  // namespace lossfunc
    }  // namespace problem
}  // namespace slam
