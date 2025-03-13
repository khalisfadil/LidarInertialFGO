#include <iostream>

#include "Problem/OptimizationProblem.hpp"

namespace slam {
    namespace problem {

        // ----------------------------------------------------------------------------
        // Constructor
        // ----------------------------------------------------------------------------

        OptimizationProblem::OptimizationProblem()
            : Problem(),
            state_vector_(std::make_shared<StateVector>()) {}

        // ----------------------------------------------------------------------------
        // addStateVariable
        // ----------------------------------------------------------------------------

        void OptimizationProblem::addStateVariable(const slam::eval::StateVariableBase::Ptr& state_var) {
            state_vars_.push_back(state_var);
        }

        // ----------------------------------------------------------------------------
        // addCostTerm
        // ----------------------------------------------------------------------------

        void OptimizationProblem::addCostTerm(slam::problem::costterm::BaseCostTerm::ConstPtr cost_term) {
            cost_terms_.push_back(cost_term);
        }

        // ----------------------------------------------------------------------------
        // getNumberOfCostTerms
        // ----------------------------------------------------------------------------

        unsigned int OptimizationProblem::getNumberOfCostTerms() const noexcept {
            return cost_terms_.size();
        }

        // ----------------------------------------------------------------------------
        // cost
        // ----------------------------------------------------------------------------

        double OptimizationProblem::cost() const noexcept {
            struct CostReducer {
                double total = 0.0;
                const std::vector<slam::problem::costterm::BaseCostTerm::ConstPtr>& terms;

                CostReducer(const std::vector<slam::problem::costterm::BaseCostTerm::ConstPtr>& t) : terms(t) {}
                CostReducer(CostReducer& other, tbb::split) : terms(other.terms) {}

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        try {
                            double cost_i = terms[i]->cost();
                            if (!std::isnan(cost_i)) {
                                total += cost_i;
                            } else {
                                std::cerr << "[OptimizationProblem::cost] Warning: Ignored NaN cost term." << std::endl;
                            }
                        } catch (const std::exception &e) {
                            std::cerr << "[OptimizationProblem::cost] Exception in cost term evaluation: " << e.what() << std::endl;
                        } catch (...) {
                            std::cerr << "[OptimizationProblem::cost] Unknown exception in cost term evaluation." << std::endl;
                        }
                    }
                }

                void join(const CostReducer& other) {
                    total += other.total;
                }
            };

            CostReducer reducer(cost_terms_);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, cost_terms_.size()), reducer);
            return reducer.total;
        }

        // ----------------------------------------------------------------------------
        // getStateVector
        // ----------------------------------------------------------------------------

        StateVector::Ptr OptimizationProblem::getStateVector() const {
            auto state_vector = std::make_shared<StateVector>();

            for (const auto &state_var : state_vars_) {
                if (!state_var->locked()) {
                    state_vector->addStateVariable(state_var);
                }
            }

            return state_vector;
        }

        // ----------------------------------------------------------------------------
        // buildGaussNewtonTerms
        // ----------------------------------------------------------------------------

        void OptimizationProblem::buildGaussNewtonTerms(
            Eigen::SparseMatrix<double>& approximate_hessian,
            Eigen::VectorXd& gradient_vector) const {
            std::vector<unsigned int> block_sizes = state_vector_->getStateBlockSizes();
            slam::blockmatrix::BlockSparseMatrix A_(block_sizes, true);
            slam::blockmatrix::BlockVector b_(block_sizes);

            tbb::parallel_for(size_t(0), cost_terms_.size(), [&](size_t i) {
                try {
                    cost_terms_[i]->buildGaussNewtonTerms(*state_vector_, &A_, &b_);
                } catch (const std::exception &e) {
                    std::cerr << "[OptimizationProblem::cost] Exception in Gauss-Newton term computation: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "[OptimizationProblem::cost] Unknown exception in Gauss-Newton term computation." << std::endl;
                }
            });

            approximate_hessian = A_.toEigen(false);
            gradient_vector = b_.toEigen();
        }
    }  // namespace problem
}  // namespace slam
