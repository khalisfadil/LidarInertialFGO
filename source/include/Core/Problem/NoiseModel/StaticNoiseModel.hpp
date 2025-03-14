#pragma once

#include <Eigen/Dense>
#include <memory>
#include <sstream>

#include "Core/Problem/NoiseModel/BaseNoiseModel.hpp"

namespace slam {
    namespace problem {
        namespace noisemodel {
            
            //-----------------------------------------------------------------------------
            /**
             * \brief StaticNoiseModel: A noise model for fixed uncertainties.
             * 
             * This model stores a static noise matrix (covariance, information, or square root information)
             * that does not change during optimization.
             * 
             * \tparam DIM The dimensionality of the noise model.
             */
            template <int DIM>
            class StaticNoiseModel : public BaseNoiseModel<DIM> {
                public:

                    //-----------------------------------------------------------------------------
                    /** \brief Convenience typedefs */
                    using Ptr = std::shared_ptr<StaticNoiseModel<DIM>>;
                    using ConstPtr = std::shared_ptr<const StaticNoiseModel<DIM>>;

                    using MatrixT = Eigen::Matrix<double, DIM, DIM>;  ///< Square matrix (DIM × DIM)
                    using VectorT = Eigen::Matrix<double, DIM, 1>;    ///< Column vector (DIM × 1)

                    //-----------------------------------------------------------------------------
                    /**
                     * \brief Factory method to create a shared instance.
                     * \param matrix The noise matrix.
                     * \param type The type of noise representation (Covariance, Information, or Sqrt Information).
                     * \return A shared pointer to the created StaticNoiseModel instance.
                     */
                    static Ptr MakeShared(const MatrixT& matrix, NoiseType type = NoiseType::COVARIANCE) {
                        return std::make_shared<StaticNoiseModel<DIM>>(matrix, type);
                    }

                    //-----------------------------------------------------------------------------
                    /**
                     * \brief Constructor.
                     * \param matrix The noise matrix.
                     * \param type The type of noise representation.
                     */
                    explicit StaticNoiseModel(const MatrixT& matrix, NoiseType type = NoiseType::COVARIANCE);

                    //-----------------------------------------------------------------------------
                    /** \brief Get the square root information matrix \( \mathbf{L} \). */
                    [[nodiscard]] MatrixT getSqrtInformation() const noexcept override;

                    //-----------------------------------------------------------------------------
                    /** \brief Compute the whitened error norm \( ||\mathbf{L} e|| \). */
                    [[nodiscard]] double getWhitenedErrorNorm(const VectorT& rawError) const noexcept override;

                    //-----------------------------------------------------------------------------
                    /** \brief Compute the whitened error vector \( e_w = \mathbf{L} e \). */
                    [[nodiscard]] VectorT whitenError(const VectorT& rawError) const noexcept override;

                private:

                    //-----------------------------------------------------------------------------
                    /** \brief Convert covariance matrix to square root information matrix. */
                    void setByCovariance(const MatrixT& matrix);

                    //-----------------------------------------------------------------------------
                    /** \brief Convert information matrix to square root information matrix. */
                    void setByInformation(const MatrixT& matrix);

                    //-----------------------------------------------------------------------------
                    /** \brief Validate and store the square root information matrix. */
                    void setBySqrtInformation(const MatrixT& matrix);

                    //-----------------------------------------------------------------------------
                    /** \brief Ensure matrix is positive definite. */
                    void assertPositiveDefiniteMatrix(const MatrixT& matrix) const;

                    //-----------------------------------------------------------------------------
                    /** \brief Stored square root information matrix. */
                    MatrixT sqrtInformation_;
                };

                // ========================== Implementation ========================== //

                //-----------------------------------------------------------------------------
                template <int DIM>
                StaticNoiseModel<DIM>::StaticNoiseModel(const MatrixT& matrix, NoiseType type) {
                    switch (type) {
                        case NoiseType::INFORMATION:
                        setByInformation(matrix);
                        break;
                        case NoiseType::SQRT_INFORMATION:
                        setBySqrtInformation(matrix);
                        break;
                        case NoiseType::COVARIANCE:
                        default:
                        setByCovariance(matrix);
                        break;
                    }
                }

                //-----------------------------------------------------------------------------
                template <int DIM>
                void StaticNoiseModel<DIM>::setByCovariance(const MatrixT& matrix) {
                    setByInformation(matrix.inverse());  // Information = inverse of covariance
                }
                
                //-----------------------------------------------------------------------------
                template <int DIM>
                void StaticNoiseModel<DIM>::setByInformation(const MatrixT& matrix) {
                    assertPositiveDefiniteMatrix(matrix);
                    Eigen::LLT<MatrixT> llt(matrix);
                    setBySqrtInformation(llt.matrixL().transpose());
                }

                //-----------------------------------------------------------------------------
                template <int DIM>
                void StaticNoiseModel<DIM>::setBySqrtInformation(const MatrixT& matrix) {
                    sqrtInformation_ = matrix;  // Ensure matrix is upper triangular if needed
                }

                //-----------------------------------------------------------------------------
                template <int DIM>
                auto StaticNoiseModel<DIM>::getSqrtInformation() const noexcept -> MatrixT {
                    return sqrtInformation_;
                }

                //-----------------------------------------------------------------------------
                template <int DIM>
                double StaticNoiseModel<DIM>::getWhitenedErrorNorm(const VectorT& rawError) const noexcept {
                    return (sqrtInformation_ * rawError).norm();
                }

                //-----------------------------------------------------------------------------
                template <int DIM>
                auto StaticNoiseModel<DIM>::whitenError(const VectorT& rawError) const noexcept -> VectorT {
                    return sqrtInformation_ * rawError;
                }

                //-----------------------------------------------------------------------------
                template <int DIM>
                void StaticNoiseModel<DIM>::assertPositiveDefiniteMatrix(const MatrixT& matrix) const {
                    Eigen::SelfAdjointEigenSolver<MatrixT> eigsolver(matrix, Eigen::EigenvaluesOnly);
                    if (eigsolver.eigenvalues().minCoeff() <= 0) {
                        std::stringstream ss;
                        ss << "Covariance matrix must be positive definite.\n"
                        << "Min eigenvalue: " << eigsolver.eigenvalues().minCoeff() << "\n"
                        << "Matrix:\n" << matrix;
                        throw std::invalid_argument(ss.str());
                    }
                }

        }  // namespace noisemodel
    }  // namespace problem
}  // namespace slam
