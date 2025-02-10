/**
 * \file DynamicNoiseModel.hpp
 * \author Sean Anderson, Alec Krawciw, ASRL
 */
#pragma once

#include <iostream>
#include <sstream>
#include <memory>
#include <Eigen/Dense>

#include "source/include/Problem/NoiseModel/BaseNoiseModel.hpp"
#include "source/include/Evaluable/Evaluable.hpp"

namespace slam {
    namespace problem {
        namespace noisemodel {
            
            //-----------------------------------------------------------------------------
            /**
             * \brief DynamicNoiseModel: A noise model for uncertainties that change with state variables.
             * 
             * This model allows the uncertainty (covariance, information, or square root information matrix) 
             * to change dynamically during the optimization process.
             * 
             * \tparam DIM The dimensionality of the noise model.
             */
            template <int DIM>
            class DynamicNoiseModel : public BaseNoiseModel<DIM> {
                public:

                    //-----------------------------------------------------------------------------
                    /** \brief Convenience typedefs */
                    using Ptr = std::shared_ptr<DynamicNoiseModel<DIM>>;
                    using ConstPtr = std::shared_ptr<const DynamicNoiseModel<DIM>>;

                    using MatrixT = Eigen::Matrix<double, DIM, DIM>;  ///< Square matrix (DIM × DIM)
                    using VectorT = Eigen::Matrix<double, DIM, 1>;    ///< Column vector (DIM × 1)
                    using MatrixTEvalPtr = typename Evaluable<MatrixT>::ConstPtr;

                    //-----------------------------------------------------------------------------
                    /**
                     * \brief Factory method to create a shared instance.
                     * \param eval A noise matrix evaluable, determined by the type parameter.
                     * \param type The type of noise representation (Covariance, Information, or Sqrt Information).
                     * \return A shared pointer to the created DynamicNoiseModel instance.
                     */
                    static Ptr MakeShared(const MatrixTEvalPtr& eval, NoiseType type = NoiseType::COVARIANCE) {
                        return std::make_shared<DynamicNoiseModel<DIM>>(eval, type);
                    }

                    //-----------------------------------------------------------------------------
                    /**
                     * \brief Constructor.
                     * \param eval A noise matrix evaluable.
                     * \param type The type of noise representation.
                     */
                    explicit DynamicNoiseModel(const MatrixTEvalPtr& eval, NoiseType type = NoiseType::COVARIANCE);

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
                    [[nodiscard]] MatrixT setByCovariance(const MatrixT& matrix) const;

                    //-----------------------------------------------------------------------------
                    /** \brief Convert information matrix to square root information matrix. */
                    [[nodiscard]] MatrixT setByInformation(const MatrixT& matrix) const;

                    //-----------------------------------------------------------------------------
                    /** \brief Validate and return the square root information matrix. */
                    [[nodiscard]] MatrixT setBySqrtInformation(const MatrixT& matrix) const;

                    //-----------------------------------------------------------------------------
                    /** \brief Validate if a matrix is positive definite. */
                    void assertPositiveDefiniteMatrix(const MatrixT& matrix) const;

                    //-----------------------------------------------------------------------------
                    /** \brief Evaluable providing the noise matrix. */
                    const MatrixTEvalPtr eval_;

                    //-----------------------------------------------------------------------------
                    /** \brief The noise representation type. */
                    NoiseType type_;
                };

                    // ========================== Implementation ========================== //

                    //-----------------------------------------------------------------------------
                    template <int DIM>
                    DynamicNoiseModel<DIM>::DynamicNoiseModel(const MatrixTEvalPtr& eval, NoiseType type)
                        : eval_(eval), type_(type) {
                        if (!eval_) {
                            throw std::invalid_argument("DynamicNoiseModel: Evaluator cannot be null.");
                        }
                    }

                    //-----------------------------------------------------------------------------
                    template <int DIM>
                    auto DynamicNoiseModel<DIM>::getSqrtInformation() const noexcept -> MatrixT {
                        const MatrixT matrix = eval_->value();
                        switch (type_) {
                            case NoiseType::INFORMATION:
                            return setByInformation(matrix);
                            case NoiseType::SQRT_INFORMATION:
                            return setBySqrtInformation(matrix);
                            case NoiseType::COVARIANCE:
                            default:
                            return setByCovariance(matrix);
                        }
                    }

                    //-----------------------------------------------------------------------------
                    template <int DIM>
                    auto DynamicNoiseModel<DIM>::setByCovariance(const MatrixT& matrix) const -> MatrixT {
                        return setByInformation(matrix.inverse());  // Information = inverse of covariance
                    }

                    //-----------------------------------------------------------------------------
                    template <int DIM>
                    auto DynamicNoiseModel<DIM>::setByInformation(const MatrixT& matrix) const -> MatrixT {
                        assertPositiveDefiniteMatrix(matrix);
                        Eigen::LLT<MatrixT> llt(matrix);
                        return setBySqrtInformation(llt.matrixL().transpose());
                    }

                    //-----------------------------------------------------------------------------
                    template <int DIM>
                    auto DynamicNoiseModel<DIM>::setBySqrtInformation(const MatrixT& matrix) const -> MatrixT {
                        return matrix;  // Ensure it's upper triangular if needed
                    }

                    //-----------------------------------------------------------------------------
                    template <int DIM>
                    double DynamicNoiseModel<DIM>::getWhitenedErrorNorm(const VectorT& rawError) const noexcept {
                        return (getSqrtInformation() * rawError).norm();
                    }

                    //-----------------------------------------------------------------------------
                    template <int DIM>
                    auto DynamicNoiseModel<DIM>::whitenError(const VectorT& rawError) const noexcept -> VectorT {
                        return getSqrtInformation() * rawError;
                    }

                    //-----------------------------------------------------------------------------
                    template <int DIM>
                    void DynamicNoiseModel<DIM>::assertPositiveDefiniteMatrix(const MatrixT& matrix) const {
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
