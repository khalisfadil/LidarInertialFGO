#pragma once

#include <memory>
#include <Eigen/Core>
#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>
#include "Core/Evaluable/Evaluable.hpp"
#include "Core/Evaluable/StateKeyJacobians.hpp"
#include "Core/Problem/CostTerm/BaseCostTerm.hpp"
#include "Core/Problem/LossFunc/BaseLossFunc.hpp"
#include "Core/Problem/NoiseModel/BaseNoiseModel.hpp"

namespace slam {
    namespace problem {
        namespace costterm {

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

                /**
                 * @brief Factory method for creating a shared pointer instance.
                 */
                static Ptr MakeShared(
                    const typename slam::eval::Evaluable<ErrorType>::ConstPtr& error_function,
                    const typename slam::problem::noisemodel::BaseNoiseModel<DIM>::ConstPtr& noise_model,
                    const typename slam::problem::lossfunc::BaseLossFunc::ConstPtr& loss_function) {
                    return std::make_shared<WeightedLeastSqCostTerm<DIM>>(error_function, noise_model, loss_function);
                }

                /**
                 * @brief Constructor for the weighted least squares cost term.
                 */
                WeightedLeastSqCostTerm(
                    const typename slam::eval::Evaluable<ErrorType>::ConstPtr& error_function,
                    const typename slam::problem::noisemodel::BaseNoiseModel<DIM>::ConstPtr& noise_model,
                    const typename slam::problem::lossfunc::BaseLossFunc::ConstPtr& loss_function)
                    : error_function_(error_function),
                      noise_model_(noise_model),
                      loss_function_(loss_function) {}

                /** @brief Destructor (override for safety). */
                ~WeightedLeastSqCostTerm() override = default;

                /** @brief Evaluates the cost. */
                [[nodiscard]] double cost() const noexcept override {
                    return loss_function_->cost(
                        noise_model_->getWhitenedErrorNorm(error_function_->evaluate()));
                }

                /** @brief Retrieves keys of related state variables. */
                void getRelatedVarKeys(KeySet& keys) const noexcept override {
                    error_function_->getRelatedVarKeys(keys);
                }

                /**
                 * @brief Builds Gauss-Newton terms (Hessian & gradient) for optimization.
                 */
                void buildGaussNewtonTerms(const StateVector& state_vec,
                    slam::blockmatrix::BlockSparseMatrix* approximate_hessian,
                    slam::blockmatrix::BlockVector* gradient_vector) const override {
                        
                    slam::eval::StateKeyJacobians jacobian_container;
                    double sqrt_weight;
                    ErrorType whitened_error = evalWeightedAndWhitened(jacobian_container, sqrt_weight);
                    ErrorType error = sqrt_weight * whitened_error;
                    auto& jacobians = jacobian_container.get();

                    for (auto& entry : jacobians) {
                        entry.second.mat *= sqrt_weight;
                    }

                    // Collect keys into a vector (no sorting)
                    std::vector<slam::eval::StateKey> keys;
                    keys.reserve(jacobians.size());
                    for (const auto& entry : jacobians) {
                        keys.push_back(entry.first);
                    }

                    struct ThreadLocalData {
                        std::vector<std::pair<unsigned int, Eigen::MatrixXd>> grad_updates;
                        std::vector<std::tuple<unsigned int, unsigned int, Eigen::MatrixXd>> hess_updates;
                    };

                    tbb::spin_mutex global_mutex;
                    tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()),
                        [&](const tbb::blocked_range<size_t>& range) {
                            ThreadLocalData local;
                            local.grad_updates.reserve(range.size());
                            local.hess_updates.reserve(range.size() * (keys.size() - range.begin() + 1) / 2); // Upper triangle

                            for (size_t i = range.begin(); i < range.end(); ++i) {
                                slam::eval::StateKeyJacobians::StateJacobianMap::const_accessor acc1;
                                jacobians.find(acc1, keys[i]);
                                const auto& key1 = acc1->first;
                                const auto& jac1 = acc1->second.mat;

                                unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);
                                Eigen::MatrixXd grad_term = -jac1.transpose() * error;
                                local.grad_updates.emplace_back(blkIdx1, std::move(grad_term));

                                for (size_t j = i; j < keys.size(); ++j) {
                                    slam::eval::StateKeyJacobians::StateJacobianMap::const_accessor acc2;
                                    jacobians.find(acc2, keys[j]);
                                    const auto& key2 = acc2->first;
                                    const auto& jac2 = acc2->second.mat;

                                    unsigned int blkIdx2 = state_vec.getStateBlockIndex(key2);
                                    unsigned int row = std::min(blkIdx1, blkIdx2);
                                    unsigned int col = std::max(blkIdx1, blkIdx2);

                                    Eigen::MatrixXd hess_term(jac1.rows(), jac2.cols());
                                    hess_term.noalias() = jac1.transpose() * jac2;
                                    local.hess_updates.emplace_back(row, col, std::move(hess_term));
                                }
                            }

                            // Merge local updates into global structures
                            tbb::spin_mutex::scoped_lock lock(global_mutex);
                            for (const auto& [idx, term] : local.grad_updates) {
                                gradient_vector->mapAt(idx).noalias() += term;
                            }
                            for (const auto& [row, col, term] : local.hess_updates) {
                                auto& entry = approximate_hessian->rowEntryAt(row, col, true);
                                entry.data.noalias() += term;
                            }
                        });
            }

            private:
                /**
                 * @brief Computes the whitened and weighted error and Jacobians.
                 */
                ErrorType evalWeightedAndWhitened(
                    slam::eval::StateKeyJacobians& jacobian_container,
                    double& sqrt_weight) const {
                    jacobian_container.clear();
                    ErrorType raw_error = error_function_->evaluate(
                        noise_model_->getSqrtInformation(), jacobian_container);
                    ErrorType whitened_error = noise_model_->whitenError(raw_error);
                    sqrt_weight = std::sqrt(loss_function_->weight(whitened_error.norm()));
                    return whitened_error;
                }

                /** @brief Error function evaluator. */
                typename slam::eval::Evaluable<ErrorType>::ConstPtr error_function_;

                /** @brief Noise model for error whitening. */
                typename slam::problem::noisemodel::BaseNoiseModel<DIM>::ConstPtr noise_model_;

                /** @brief Loss function to downweight large residuals. */
                typename slam::problem::lossfunc::BaseLossFunc::ConstPtr loss_function_;
            };

        }  // namespace costterm
    }  // namespace problem
}  // namespace slam