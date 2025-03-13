#pragma once

#include <cmath>

#include "Problem/LossFunc/BaseLossFunc.hpp"

namespace slam {
    namespace problem {
        namespace lossfunc {
        
        // -----------------------------------------------------------------------------
        /**
         * \brief Huber Loss Function for robust estimation.
         * 
         * The Huber loss function is a piecewise function that reduces the 
         * influence of large residuals while preserving a quadratic behavior 
         * for small errors. It is widely used in SLAM, sensor fusion, and 
         * robust optimization.
         */
        class HuberLossFunc : public BaseLossFunc {
            public:
                // -----------------------------------------------------------------------------
                /** \brief Convenience typedefs */
                using Ptr = std::shared_ptr<HuberLossFunc>;
                using ConstPtr = std::shared_ptr<const HuberLossFunc>;

                // -----------------------------------------------------------------------------
                /**
                 * \brief Factory method to create a shared instance of HuberLossFunc.
                 * \param k Robustness threshold (number of standard deviations).
                 * \return A shared pointer to the created HuberLossFunc instance.
                 */
                static Ptr MakeShared(double k) {
                    return std::make_shared<HuberLossFunc>(k);
                }

                // -----------------------------------------------------------------------------
                /**
                 * \brief Constructor for Huber Loss Function.
                 * \param k The threshold parameter controlling robustness.
                 */
                explicit HuberLossFunc(double k) noexcept : k_(k), half_k_(0.5 * k) {}

                // -----------------------------------------------------------------------------
                /**
                 * \brief Computes the cost function value.
                 * \param whitened_error_norm The whitened error norm.
                 * \return The computed cost.
                 */
                [[nodiscard]] double cost(double whitened_error_norm) const noexcept override {
                    double e2 = whitened_error_norm * whitened_error_norm;
                    double abse = std::abs(whitened_error_norm);
                    return (abse <= k_) ? 0.5 * e2 : k_ * (abse - half_k_);
                }

                // -----------------------------------------------------------------------------
                /**
                 * \brief Computes the weight for iteratively reweighted least squares (IRLS).
                 * \param whitened_error_norm The whitened error norm.
                 * \return The computed weight.
                 */
                [[nodiscard]] double weight(double whitened_error_norm) const noexcept override {
                    double abse = std::abs(whitened_error_norm);
                    return (abse <= k_) ? 1.0 : k_ / abse;
                }

            private:

                // -----------------------------------------------------------------------------
                /** \brief Huber threshold constant \( k \) */
                const double k_;

                // -----------------------------------------------------------------------------
                /** \brief Precomputed \( 0.5 \times k \) to avoid redundant computation */
                const double half_k_;
            };
        }  // namespace lossfunc
    }  // namespace problem
}  // namespace slam
