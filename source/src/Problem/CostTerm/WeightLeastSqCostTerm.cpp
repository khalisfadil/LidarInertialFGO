#include "source/include/Problem/CostTerm/WeightLeastSqCostTerm.hpp"

namespace slam {
    namespace problem {
        namespace costterm {

            // ----------------------------------------------------------------------------
            // MakeShared
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto WeightedLeastSqCostTerm<DIM>::MakeShared(
                const slam::eval::Evaluable<ErrorType>::ConstPtr &error_function,
                const slam::problem::noisemodel::BaseNoiseModel<DIM>::ConstPtr &noise_model,
                const slam::problem::lossfunc::BaseLossFunc::ConstPtr &loss_function) -> Ptr {
                return std::make_shared<WeightedLeastSqCostTerm<DIM>>(error_function, noise_model, loss_function);
            }
            
            // ----------------------------------------------------------------------------
            // WeightedLeastSqCostTerm
            // ----------------------------------------------------------------------------

            template <int DIM>
            WeightedLeastSqCostTerm<DIM>::WeightedLeastSqCostTerm(
                const slam::eval::Evaluable<ErrorType>::ConstPtr &error_function,
                const slam::problem::noisemodel::BaseNoiseModel<DIM>::ConstPtr &noise_model,
                const slam::problem::lossfunc::BaseLossFunc::ConstPtr &loss_function)
                : error_function_(error_function),
                noise_model_(noise_model),
                loss_function_(loss_function) {}

            template <int DIM>
            double WeightedLeastSqCostTerm<DIM>::cost() const {
                return loss_function_->cost(
                    noise_model_->getWhitenedErrorNorm(error_function_->evaluate()));
            }

            // ----------------------------------------------------------------------------
            // getRelatedVarKeys
            // ----------------------------------------------------------------------------

            template <int DIM>
            void WeightedLeastSqCostTerm<DIM>::getRelatedVarKeys(KeySet &keys) const {
                error_function_->getRelatedVarKeys(keys);
            }

            // ----------------------------------------------------------------------------
            // evalWeightedAndWhitened
            // ----------------------------------------------------------------------------
            /**
             * @brief Computes the whitened and weighted error and Jacobians.
             * This function performs:
             * 1. **Raw error evaluation**
             * 2. **Whitening with the noise model**
             * 3. **Applying a loss function weight**
             * 4. **Scaling Jacobians accordingly**
             */

            template <int DIM>
            typename WeightedLeastSqCostTerm<DIM>::ErrorType 
            WeightedLeastSqCostTerm<DIM>::evalWeightedAndWhitened(slam::eval::StateKeyJacobians &jacobian_container) const {
                // Reset Jacobian storage
                jacobian_container.clear();

                // Evaluate raw error and collect Jacobians
                ErrorType raw_error = error_function_->evaluate(
                    noise_model_->getSqrtInformation(), jacobian_container);

                // Whiten the error using the noise model
                ErrorType whitened_error = noise_model_->whitenError(raw_error);

                // Compute loss function weight
                double sqrt_weight = std::sqrt(loss_function_->weight(whitened_error.norm()));

                // Scale Jacobians
                auto &jacobians = jacobian_container.get();
                for (auto &entry : jacobians) {
                    entry.second *= sqrt_weight;
                }

                // Return the final weighted and whitened error
                return sqrt_weight * whitened_error;
            }

            // ----------------------------------------------------------------------------
            // buildGaussNewtonTerms
            // ----------------------------------------------------------------------------

            template <int DIM>
            void WeightedLeastSqCostTerm<DIM>::buildGaussNewtonTerms(
                const StateVector &state_vec,
                slam::blockmatrix::BlockSparseMatrix *approximate_hessian,
                slam::blockmatrix::BlockVector *gradient_vector) const {
                
                // Compute the weighted error and Jacobians
                slam::eval::StateKeyJacobians jacobian_container;
                ErrorType error = evalWeightedAndWhitened(jacobian_container);
                const auto &jacobians = jacobian_container.get();

                // Use a local mutex instead of a global one to reduce contention
                tbb::spin_mutex local_mutex;

                // Parallelize only the outer loop to avoid excessive thread spawning
                tbb::parallel_for(tbb::blocked_range<size_t>(0, jacobians.size()), [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        auto it1 = std::next(jacobians.begin(), i);
                        const auto &key1 = it1->first;
                        const auto &jac1 = it1->second;

                        unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);
                        Eigen::MatrixXd newGradTerm = -jac1.transpose() * error;

                        {
                            tbb::spin_mutex::scoped_lock lock(local_mutex);
                            gradient_vector->mapAt(blkIdx1) += newGradTerm;
                        }

                        // Reduce parallel nesting by sequentially iterating over inner elements
                        for (size_t j = i; j < jacobians.size(); ++j) {
                            auto it2 = std::next(jacobians.begin(), j);
                            const auto &key2 = it2->first;
                            const auto &jac2 = it2->second;

                            unsigned int blkIdx2 = state_vec.getStateBlockIndex(key2);
                            unsigned int row = std::min(blkIdx1, blkIdx2);
                            unsigned int col = std::max(blkIdx1, blkIdx2);

                            Eigen::MatrixXd newHessianTerm = jac1.transpose() * jac2;

                            BlockSparseMatrix::BlockRowEntry &entry = approximate_hessian->rowEntryAt(row, col, true);
                            {
                                tbb::spin_mutex::scoped_lock lock(local_mutex);
                                entry.data += newHessianTerm;
                            }
                        }
                    }
                });
            }

            // Explicit template instantiation
            template class WeightedLeastSqCostTerm<1>;
            template class WeightedLeastSqCostTerm<2>;
            template class WeightedLeastSqCostTerm<3>;

        }  // namespace costterm
    }  // namespace problem
}  // namespace slam
