#include "source/include/Problem/SlidingWindowFilter.hpp"

namespace slam {
    namespace problem {

        // -----------------------------------------------------------------------------
        // MakeShared
        // -----------------------------------------------------------------------------

        SlidingWindowFilter::Ptr SlidingWindowFilter::MakeShared(unsigned int num_threads) {
            return std::make_shared<SlidingWindowFilter>(num_threads);
        }

        // -----------------------------------------------------------------------------
        // SlidingWindowFilter
        // -----------------------------------------------------------------------------

        SlidingWindowFilter::SlidingWindowFilter(unsigned int num_threads)
            : num_threads_(num_threads),
            active_state_vector_(std::make_shared<StateVector>()),
            marginalize_state_vector_(std::make_shared<StateVector>()),
            state_vector_(std::make_shared<StateVector>()) {}

        // -----------------------------------------------------------------------------
        // addStateVariable
        // -----------------------------------------------------------------------------

        void SlidingWindowFilter::addStateVariable(const slam::eval::StateVariableBase::Ptr& variable) {
            addStateVariable(std::vector<slam::eval::StateVariableBase::Ptr>{variable});
        }

        // -----------------------------------------------------------------------------
        // addStateVariable
        // -----------------------------------------------------------------------------

        void SlidingWindowFilter::addStateVariable(const std::vector<slam::eval::StateVariableBase::Ptr>& variables) {
            for (const auto& variable : variables) {
                const auto res = variables_.try_emplace(variable->key(), variable, false);
                if (!res.second) throw std::runtime_error("[SlidingWindowFilter::addStateVariable] Duplicate variable key detected.");
                variable_queue_.emplace_back(variable->key());
                related_var_keys_.try_emplace(variable->key(), KeySet{variable->key()});
            }
        }

        // -----------------------------------------------------------------------------
        // marginalizeVariable
        // -----------------------------------------------------------------------------

        void SlidingWindowFilter::marginalizeVariable(const slam::eval::StateVariableBase::Ptr& variable) {
            marginalizeVariable(std::vector<slam::eval::StateVariableBase::Ptr>{variable});
        }

        // -----------------------------------------------------------------------------
        // marginalizeVariable
        // -----------------------------------------------------------------------------

        void SlidingWindowFilter::marginalizeVariable(const std::vector<slam::eval::StateVariableBase::Ptr>& variables) {
            if (variables.empty()) return;

            // Mark variables for marginalization
            for (const auto& variable : variables) {
                variables_.at(variable->key()).marginalize = true;
            }

            // Create state vectors for fixed and active variables
            StateVector fixed_state_vector, state_vector;
            std::vector<slam::eval::StateKey> to_remove;
            bool fixed = true;

            // Identify variables to be marginalized
            for (const auto& key : variable_queue_) {
                const auto& var = variables_.at(key);
                const auto& related_keys = related_var_keys_.at(key);

                if (std::all_of(related_keys.begin(), related_keys.end(), [this](const slam::eval::StateKey& key) {
                        return variables_.at(key).marginalize;
                    })) {
                    if (!fixed) throw std::runtime_error("[SlidingWindowFilter::marginalizeVariable] Fixed variables must be at the beginning.");
                    fixed_state_vector.addStateVariable(var.variable);
                    to_remove.emplace_back(key);
                } else {
                    fixed = false;
                }
                state_vector.addStateVariable(var.variable);
            }

            // Process cost terms
            tbb::concurrent_vector<slam::problem::costterm::BaseCostTerm::ConstPtr> active_cost_terms;
            slam::blockmatrix::BlockSparseMatrix A_(state_vector.getStateBlockSizes(), true);
            slam::blockmatrix::BlockVector b_(state_vector.getStateBlockSizes());

            tbb::parallel_for(size_t(0), cost_terms_.size(), [&](size_t c) {
                KeySet keys;
                cost_terms_[c]->getRelatedVarKeys(keys);

                if (std::all_of(keys.begin(), keys.end(), [this](const slam::eval::StateKey& key) {
                        return variables_.at(key).marginalize;
                    })) {
                    cost_terms_[c]->buildGaussNewtonTerms(state_vector, &A_, &b_);
                } else {
                    active_cost_terms.emplace_back(cost_terms_[c]);
                }
            });

            cost_terms_.assign(active_cost_terms.begin(), active_cost_terms.end());

            // Convert sparse matrix to Eigen dense format
            Eigen::MatrixXd A(A_.toEigen(false).selfadjointView<Eigen::Upper>());
            Eigen::VectorXd b(b_.toEigen());

            // Add cached terms
            if (!fixed_A_.isZero()) {
                A.topLeftCorner(fixed_A_.rows(), fixed_A_.cols()) += fixed_A_;
                b.head(fixed_b_.size()) += fixed_b_;
            }

            // Marginalization step
            size_t fixed_state_size = fixed_state_vector.getStateSize();
            if (fixed_state_size > 0) {
                Eigen::MatrixXd A00 = A.topLeftCorner(fixed_state_size, fixed_state_size);
                Eigen::MatrixXd A10 = A.bottomLeftCorner(A.rows() - fixed_state_size, fixed_state_size);
                Eigen::MatrixXd A11 = A.bottomRightCorner(A.rows() - fixed_state_size, A.cols() - fixed_state_size);
                Eigen::VectorXd b0 = b.head(fixed_state_size);
                Eigen::VectorXd b1 = b.tail(b.size() - fixed_state_size);

                fixed_A_ = A11 - A10 * A00.llt().solve(A10.transpose());
                fixed_b_ = b1 - A10 * A00.inverse() * b0;
            } else {
                fixed_A_ = A;
                fixed_b_ = b;
            }

            // Remove fixed variables from queue
            for (const auto& key : to_remove) {
                related_var_keys_.erase(key);
                variables_.erase(key);
                if (variable_queue_.empty() || variable_queue_.front() != key)
                    throw std::runtime_error("[SlidingWindowFilter::marginalizeVariable] Variable queue is inconsistent.");
                variable_queue_.pop_front();
            }
        }

        // -----------------------------------------------------------------------------
        // addCostTerm
        // -----------------------------------------------------------------------------

        void SlidingWindowFilter::addCostTerm(slam::problem::costterm::BaseCostTerm::ConstPtr cost_term) {
            cost_terms_.emplace_back(cost_term);

            KeySet related_keys;
            cost_term->getRelatedVarKeys(related_keys);
            for (const auto& key : related_keys) {
                related_var_keys_.at(key).insert(related_keys.begin(), related_keys.end());
            }
        }

        // -----------------------------------------------------------------------------
        // cost
        // -----------------------------------------------------------------------------

        double SlidingWindowFilter::cost() const noexcept {
            double total_cost = 0.0;
            tbb::parallel_for(size_t(0), cost_terms_.size(), [&](size_t i) {
                double cost_i = cost_terms_[i]->cost();
                if (!std::isnan(cost_i)) {
                    total_cost += cost_i;
                }
            });
            return total_cost;
        }

        // -----------------------------------------------------------------------------
        // getNumberOfCostTerms
        // -----------------------------------------------------------------------------

        unsigned int SlidingWindowFilter::getNumberOfCostTerms() const noexcept {
            return cost_terms_.size();
        }

        // -----------------------------------------------------------------------------
        // CgetNumberOfVariables
        // -----------------------------------------------------------------------------

        unsigned int SlidingWindowFilter::getNumberOfVariables() const noexcept {
            return variable_queue_.size();
        }

        // -----------------------------------------------------------------------------
        // getStateVector
        // -----------------------------------------------------------------------------

        StateVector::Ptr SlidingWindowFilter::getStateVector() const {
            auto state_vector = std::make_shared<StateVector>();

            for (const auto& key : variable_queue_) {
                state_vector->addStateVariable(variables_.at(key).variable);
            }

            return state_vector;
        }

        // -----------------------------------------------------------------------------
        // buildGaussNewtonTerms
        // -----------------------------------------------------------------------------

        void SlidingWindowFilter::buildGaussNewtonTerms(Eigen::SparseMatrix<double>& approximate_hessian,
                                                        Eigen::VectorXd& gradient_vector) const {
            std::vector<unsigned int> block_sizes = state_vector_->getStateBlockSizes();
            slam::blockmatrix::BlockSparseMatrix A_(block_sizes, true);
            slam::blockmatrix::BlockVector b_(block_sizes);

            tbb::parallel_for(size_t(0), cost_terms_.size(), [&](size_t i) {
                cost_terms_[i]->buildGaussNewtonTerms(*state_vector_, &A_, &b_);
            });

            Eigen::MatrixXd A(A_.toEigen(false).selfadjointView<Eigen::Upper>());
            Eigen::VectorXd b(b_.toEigen());

            if (!fixed_A_.isZero()) {
                A.topLeftCorner(fixed_A_.rows(), fixed_A_.cols()) += fixed_A_;
                b.head(fixed_b_.size()) += fixed_b_;
            }

            approximate_hessian = A.sparseView();
            gradient_vector = b;
        }

    }  // namespace problem
}  // namespace slam
