#pragma once


#include <cmath>

#include "source/include/Problem/LossFunc/BaseLossFunc.hpp"

namespace slam {
    namespace problem {
        namespace lossfunc {
            
            // -----------------------------------------------------------------------------
            /**
             * \brief Cauchy loss function for robust estimation.
             * 
             * This function reduces the influence of outliers while maintaining 
             * good performance for small residuals. The parameter \( k \) defines 
             * the threshold based on the number of standard deviations (typically 1-3).
             */
            class CauchyLossFunc : public BaseLossFunc {
                public:

                    // -----------------------------------------------------------------------------
                    /** \brief Convenience typedefs */
                    using Ptr = std::shared_ptr<CauchyLossFunc>;
                    using ConstPtr = std::shared_ptr<const CauchyLossFunc>;

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Factory method to create a shared instance of CauchyLossFunc.
                     * \param k The threshold parameter for the loss function.
                     * \return A shared pointer to the created CauchyLossFunc instance.
                     */
                    static Ptr MakeShared(double k) {
                        return std::make_shared<CauchyLossFunc>(k);
                    }

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Constructor to initialize Cauchy loss function.
                     * \param k The threshold parameter controlling the influence of large residuals.
                     */
                    explicit CauchyLossFunc(double k) noexcept : k_inv_(1.0 / k), k2_(k * k) {}

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Computes the cost function value.
                     * \param whitened_error_norm The whitened error norm.
                     * \return The computed cost.
                     */
                    [[nodiscard]] double cost(double whitened_error_norm) const noexcept override {
                        double e_div_k = whitened_error_norm * k_inv_;
                        return 0.5 * k2_ * std::log1p(e_div_k * e_div_k);  // std::log1p(x) for better numerical stability
                    }

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Computes the weight for iteratively reweighted least squares (IRLS).
                     * \param whitened_error_norm The whitened error norm.
                     * \return The computed weight.
                     */
                    [[nodiscard]] double weight(double whitened_error_norm) const noexcept override {
                        double e_div_k = whitened_error_norm * k_inv_;
                        return 1.0 / (1.0 + e_div_k * e_div_k);
                    }

                private:

                    // -----------------------------------------------------------------------------
                    /** \brief Precomputed inverse of k for efficiency */
                    const double k_inv_;

                    // -----------------------------------------------------------------------------
                    /** \brief Precomputed \( k^2 \) to avoid redundant multiplication */
                    const double k2_;
            };

        }  // namespace lossfunc
    }  // namespace problem
}  // namespace slam
