#pragma once


#include <cmath>
#include <limits>

#include "source/include/Problem/LossFunc/BaseLossFunc.hpp"

namespace slam {
    namespace problem {
        namespace lossfunc {

            // -----------------------------------------------------------------------------
            /**
             * \brief L1 (Absolute) Loss Function for robust estimation.
             * 
             * The L1 loss function provides strong robustness to outliers by penalizing
             * errors linearly rather than quadratically. This makes it ideal for
             * SLAM, sensor fusion, and robust optimization tasks.
             */
            class L1LossFunc : public BaseLossFunc {
                public:

                    // -----------------------------------------------------------------------------
                    /** \brief Convenience typedefs */
                    using Ptr = std::shared_ptr<L1LossFunc>;
                    using ConstPtr = std::shared_ptr<const L1LossFunc>;

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Factory method to create a shared instance of L1LossFunc.
                     * \return A shared pointer to the created L1LossFunc instance.
                     */
                    static Ptr MakeShared() {
                        return std::make_shared<L1LossFunc>();
                    }

                    // -----------------------------------------------------------------------------
                    /** \brief Default constructor */
                    L1LossFunc() = default;

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Computes the cost function value.
                     * \param whitened_error_norm The whitened error norm.
                     * \return The computed cost.
                     */
                    [[nodiscard]] double cost(double whitened_error_norm) const noexcept override {
                        return std::abs(whitened_error_norm);
                    }

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Computes the weight for iteratively reweighted least squares (IRLS).
                     * \param whitened_error_norm The whitened error norm.
                     * \return The computed weight.
                     */
                    [[nodiscard]] double weight(double whitened_error_norm) const noexcept override {
                        return 1.0 / (std::abs(whitened_error_norm) + epsilon_);
                    }

                private:
                
                    // -----------------------------------------------------------------------------
                    static constexpr double epsilon_ = 1e-10;  // Small constant for stability
            };
        }  // namespace lossfunc
    }  // namespace problem
}  // namespace slam
