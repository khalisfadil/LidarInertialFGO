#pragma once

#include <memory>

namespace slam {
    namespace problem {
        namespace lossfunc {
            
            // -----------------------------------------------------------------------------
            /**
             * \brief Base class for loss functions.
             * 
             * A loss function must implement both the cost and weight functions.
             * Example: The least-square L2 loss function has:
             * - Cost: \( e^2 \)
             * - Weight: \( 1 \)
             */
            class BaseLossFunc {
                public:
                    // -----------------------------------------------------------------------------
                    /** \brief Convenience typedefs */
                    using Ptr = std::shared_ptr<BaseLossFunc>;
                    using ConstPtr = std::shared_ptr<const BaseLossFunc>;

                    // -----------------------------------------------------------------------------
                    /** \brief Default virtual destructor */
                    virtual ~BaseLossFunc() = default;

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Computes the cost of a given whitened error norm.
                     * \param whitened_error_norm The whitened error norm.
                     * \return The computed loss cost.
                     */
                    [[nodiscard]] virtual double cost(double whitened_error_norm) const noexcept = 0;

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Computes the weight for iteratively reweighted least squares (IRLS).
                     * \param whitened_error_norm The whitened error norm.
                     * \return The computed weight (influence function divided by error).
                     */
                    [[nodiscard]] virtual double weight(double whitened_error_norm) const noexcept = 0;
            };

        }  // namespace lossfunc
    }  // namespace problem
}  // namespace slam
