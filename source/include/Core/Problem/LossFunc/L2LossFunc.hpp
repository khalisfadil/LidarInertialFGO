#pragma once

#include "Core/Problem/LossFunc/BaseLossFunc.hpp"

namespace slam {
    namespace problem {
        namespace lossfunc {

            /**
             * \brief L2 (Least Squares) Loss Function.
             * 
             * The L2 loss function is the standard squared error loss used in
             * optimization and estimation problems. It is convex, differentiable,
             * and effective for Gaussian-distributed residuals but sensitive to outliers.
             */
            class L2LossFunc : public BaseLossFunc {
                public:

                    // -----------------------------------------------------------------------------
                    /** \brief Convenience typedefs */
                    using Ptr = std::shared_ptr<L2LossFunc>;
                    using ConstPtr = std::shared_ptr<const L2LossFunc>;

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Factory method to create a shared instance of L2LossFunc.
                     * \return A shared pointer to the created L2LossFunc instance.
                     */
                    static Ptr MakeShared() {
                        return std::make_shared<L2LossFunc>();
                    }

                    // -----------------------------------------------------------------------------
                    /** \brief Default constructor */
                    L2LossFunc() = default;

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Computes the cost function value.
                     * \param whitened_error_norm The whitened error norm.
                     * \return The computed cost.
                     */
                    [[nodiscard]] double cost(double whitened_error_norm) const noexcept override {
                        return 0.5 * whitened_error_norm * whitened_error_norm;
                    }

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Computes the weight for iteratively reweighted least squares (IRLS).
                     * \param whitened_error_norm The whitened error norm.
                     * \return The computed weight.
                     */
                    [[nodiscard]] double weight(double /*whitened_error_norm*/) const noexcept override {
                        return 1.0;  // All residuals are weighted equally
                    }
            };

        }  // namespace lossfunc
    }  // namespace problem
}  // namespace slam
