#pragma once

#include <Eigen/Dense>
#include <memory>

namespace slam {
    namespace problem {
        namespace noisemodel {
            
            // -----------------------------------------------------------------------------
            /**
             * \brief Enumeration of ways to define noise models.
             * 
             * - **COVARIANCE**: Uses covariance matrix \( \mathbf{\Sigma} \).
             * - **INFORMATION**: Uses information matrix \( \mathbf{\Lambda} \).
             * - **SQRT_INFORMATION**: Uses square root information matrix \( \mathbf{L} \).
             */
            enum class NoiseType { COVARIANCE, INFORMATION, SQRT_INFORMATION };

            // -----------------------------------------------------------------------------
            /**
             * \brief Base class for noise models in estimation and SLAM.
             * 
             * This class defines the interface for noise models, allowing different
             * representations of uncertainty. Derived classes must implement methods
             * to provide the square root information matrix and whitened error norms.
             * 
             * \tparam DIM The dimensionality of the noise model.
             */
            template <int DIM>
            
            class BaseNoiseModel {
                public:

                    // -----------------------------------------------------------------------------
                    /** \brief Convenience typedefs */
                    using Ptr = std::shared_ptr<BaseNoiseModel<DIM>>;
                    using ConstPtr = std::shared_ptr<const BaseNoiseModel<DIM>>;

                    using MatrixT = Eigen::Matrix<double, DIM, DIM>;  ///< Square matrix of size DIM
                    using VectorT = Eigen::Matrix<double, DIM, 1>;    ///< Column vector of size DIM

                    // -----------------------------------------------------------------------------
                    /** \brief Virtual destructor */
                    virtual ~BaseNoiseModel() = default;

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Get the square root information matrix \( \mathbf{L} \).
                     * \return The square root information matrix \( \mathbf{L} \) such that \( \mathbf{L}^T \mathbf{L} = \mathbf{\Lambda} \).
                     */
                    [[nodiscard]] virtual MatrixT getSqrtInformation() const noexcept = 0;

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Compute the whitened error norm, defined as:
                     * 
                     * \[
                     * ||e_w|| = \sqrt{e^T \mathbf{\Lambda} e} = ||\mathbf{L} e||
                     * \]
                     * 
                     * \param rawError The unwhitened error vector \( e \).
                     * \return The norm of the whitened error.
                     */
                    [[nodiscard]] virtual double getWhitenedErrorNorm(const VectorT& rawError) const noexcept = 0;

                    // -----------------------------------------------------------------------------
                    /**
                     * \brief Compute the whitened error vector:
                     * 
                     * \[
                     * e_w = \mathbf{L} e
                     * \]
                     * 
                     * \param rawError The unwhitened error vector \( e \).
                     * \return The whitened error vector \( e_w \).
                     */
                    [[nodiscard]] virtual VectorT whitenError(const VectorT& rawError) const noexcept = 0;
            };
        }  // namespace noisemodel
    }  // namespace problem
}  // namespace slam
