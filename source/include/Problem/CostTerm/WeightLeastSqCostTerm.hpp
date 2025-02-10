#pragma once

#include <vector>
#include <memory>
#include <Eigen/Core>
#include <iostream>
#include <unordered_set>
#include <optional>
#include <stdexcept>
#include <typeinfo>
#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>

#include "source/include/Evaluable/Evaluable.hpp"
#include "source/include/Evaluable/StateKeyJacobians.hpp"
#include "source/include/Problem/CostTerm/BaseCostTerm.hpp"
#include "source/include/Problem/LossFunc/BaseLossFunc.hpp"
#include "source/include/Problem/NoiseModel/BaseNoiseModel.hpp"

namespace slam {
    namespace problem {
        namespace costterm {

            // -----------------------------------------------------------------------------
            /**
             * @class WeightedLeastSqCostTerm
             * @brief Implements a weighted least squares cost term for optimization.
             *
             * This class applies a weighted least squares formulation:
             *     cost = loss(sqrt(e^T * cov^{-1} * e))
             * where `e` is the measurement error.
             *
             * @tparam DIM Dimensionality of the error term.
             */
            template <int DIM>
            class WeightedLeastSqCostTerm : public BaseCostTerm {
            public:
                using Ptr = std::shared_ptr<WeightedLeastSqCostTerm<DIM>>;
                using ConstPtr = std::shared_ptr<const WeightedLeastSqCostTerm<DIM>>;
                using ErrorType = Eigen::Matrix<double, DIM, 1>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method for creating a shared pointer instance.
                 */
                static Ptr MakeShared(
                    const slam::eval::Evaluable<ErrorType>::ConstPtr &error_function,
                    const slam::problem::noisemodel::BaseNoiseModel<DIM>::ConstPtr &noise_model,
                    const slam::problem::lossfunc::BaseLossFunc::ConstPtr &loss_function);
                
                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for the weighted least squares cost term.
                 */
                WeightedLeastSqCostTerm(
                    const slam::eval::Evaluable<ErrorType>::ConstPtr &error_function,
                    const slam::problem::noisemodel::BaseNoiseModel<DIM>::ConstPtr &noise_model,
                    const slam::problem::lossfunc::BaseLossFunc::ConstPtr &loss_function);

                // -----------------------------------------------------------------------------
                /** @brief Destructor (override for safety). */
                ~WeightedLeastSqCostTerm() override = default;

                // -----------------------------------------------------------------------------
                /** @brief Evaluates the cost. */
                [[nodiscard]] double cost() const override;

                // -----------------------------------------------------------------------------
                /** @brief Retrieves keys of related state variables. */
                void getRelatedVarKeys(KeySet &keys) const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Builds Gauss-Newton terms (Hessian & gradient) for optimization.
                 */
                void buildGaussNewtonTerms(
                    const StateVector &state_vec,
                    slam::blockmatrix::BlockSparseMatrix *approximate_hessian,
                    slam::blockmatrix::BlockVector *gradient_vector) const override;

            private:
                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes the whitened and weighted error and Jacobians.
                 */
                ErrorType evalWeightedAndWhitened(slam::eval::StateKeyJacobians &jacobian_container) const;

                // -----------------------------------------------------------------------------
                /** @brief Error function evaluator. */
                slam::eval::Evaluable<ErrorType>::ConstPtr error_function_;

                // -----------------------------------------------------------------------------
                /** @brief Noise model for error whitening. */
                slam::problem::noisemodel::BaseNoiseModel<DIM>::ConstPtr noise_model_;

                // -----------------------------------------------------------------------------
                /** @brief Loss function to downweight large residuals. */
                slam::problem::lossfunc::BaseLossFunc::ConstPtr loss_function_;
            };

        }  // namespace costterm
    }  // namespace problem
}  // namespace slam
