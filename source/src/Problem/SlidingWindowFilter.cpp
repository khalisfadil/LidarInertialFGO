#include "Problem/SlidingWindowFilter.hpp"

namespace slam {
    namespace problem {

        // -----------------------------------------------------------------------------
        // MakeShared
        // -----------------------------------------------------------------------------

        SlidingWindowFilter::Ptr SlidingWindowFilter::MakeShared() {
            return std::make_shared<SlidingWindowFilter>();
        }

        // -----------------------------------------------------------------------------
        // SlidingWindowFilter
        // -----------------------------------------------------------------------------

        SlidingWindowFilter::SlidingWindowFilter()
            : Problem(), // ✅ Correctly initialize the base class 
            active_state_vector_(std::make_shared<StateVector>()),
            marginalize_state_vector_(std::make_shared<StateVector>()),
            state_vector_(std::make_shared<StateVector>()) {}

        // -----------------------------------------------------------------------------
        // addStateVariable (Single Variable)
        // -----------------------------------------------------------------------------

        void SlidingWindowFilter::addStateVariable(const slam::eval::StateVariableBase::Ptr& variable) {
            VariableMap::accessor accessor;
            if (!variables_.insert(accessor, {variable->key(), Variable(variable, false)})) {
                throw std::runtime_error("[SlidingWindowFilter::addStateVariable] Duplicate variable key detected.");
            }
            variable_queue_.push_back(variable->key());
            RelatedVarKeysMap::accessor rel_accessor;
            related_var_keys_.insert(rel_accessor, variable->key());
            rel_accessor->second.insert(variable->key());
        }

        // -----------------------------------------------------------------------------
        // addStateVariable (Vector of Variables)
        // -----------------------------------------------------------------------------

        void SlidingWindowFilter::addStateVariable(const std::vector<slam::eval::StateVariableBase::Ptr>& variables) {
            for (const auto& variable : variables) {
                VariableMap::accessor accessor;
                if (!variables_.insert(accessor, {variable->key(), Variable(variable, false)})) {
                    throw std::runtime_error("[SlidingWindowFilter::addStateVariable] Duplicate variable key detected.");
                }
                variable_queue_.push_back(variable->key());
                RelatedVarKeysMap::accessor rel_accessor;
                related_var_keys_.insert(rel_accessor, variable->key());
                rel_accessor->second.insert(variable->key());
            }
        }

        // -----------------------------------------------------------------------------
        // marginalizeVariable (Single Variable)
        // -----------------------------------------------------------------------------

        void SlidingWindowFilter::marginalizeVariable(const slam::eval::StateVariableBase::Ptr& variable) {
            marginalizeVariable(std::vector<slam::eval::StateVariableBase::Ptr>{variable});
        }

        // -----------------------------------------------------------------------------
        // marginalizeVariable (Vector of Variables)
        // -----------------------------------------------------------------------------

        void SlidingWindowFilter::marginalizeVariable(const std::vector<slam::eval::StateVariableBase::Ptr>& variables) {
            if (variables.empty()) return;

            // Step 1: Cache marginalized variables for fast lookups
            std::unordered_set<slam::eval::StateKey> marginalized_keys;
            for (const auto& variable : variables) {
                VariableMap::accessor accessor;
                if (variables_.find(accessor, variable->key())) {
                    accessor->second.marginalize = true;
                    marginalized_keys.insert(variable->key());
                }
            }

            // Step 2: Identify variables to be marginalized
            StateVector fixed_state_vector, state_vector;
            std::vector<slam::eval::StateKey> to_remove;
            bool fixed = true;

            for (const auto& key : variable_queue_) {
                VariableMap::const_accessor var_accessor;
                RelatedVarKeysMap::const_accessor rel_accessor;

                if (!variables_.find(var_accessor, key) || !related_var_keys_.find(rel_accessor, key)) {
                    continue;
                }

                if (std::all_of(rel_accessor->second.begin(), rel_accessor->second.end(), [&](const slam::eval::StateKey& key) {
                    return marginalized_keys.count(key) > 0;
                })) {
                    if (!fixed) throw std::runtime_error("[SlidingWindowFilter::marginalizeVariable] Fixed variables must be at the beginning.");
                    fixed_state_vector.addStateVariable(var_accessor->second.variable);
                    to_remove.emplace_back(key);
                } else {
                    fixed = false;
                }
                state_vector.addStateVariable(var_accessor->second.variable);
            }

            // Step 3: Process cost terms
            tbb::concurrent_vector<slam::problem::costterm::BaseCostTerm::ConstPtr> active_cost_terms;
            slam::blockmatrix::BlockSparseMatrix A_(state_vector.getStateBlockSizes(), true);
            slam::blockmatrix::BlockVector b_(state_vector.getStateBlockSizes());

            tbb::parallel_for(size_t(0), cost_terms_.size(), [&](size_t c) {
                KeySet keys;
                cost_terms_[c]->getRelatedVarKeys(keys);

                if (std::all_of(keys.begin(), keys.end(), [&](const slam::eval::StateKey& key) {
                    return marginalized_keys.count(key) > 0;
                })) {
                    cost_terms_[c]->buildGaussNewtonTerms(state_vector, &A_, &b_);
                } else {
                    active_cost_terms.push_back(cost_terms_[c]);
                }
            });

            // Step 4: Update cost terms safely
            cost_terms_.clear();
            cost_terms_.assign(active_cost_terms.begin(), active_cost_terms.end());

            // Step 5: Perform marginalization (Schur Complement)
            Eigen::MatrixXd Aupper(A_.toEigen(false));
            Eigen::MatrixXd A(Aupper.selfadjointView<Eigen::Upper>());
            Eigen::VectorXd b(b_.toEigen());

            if (!fixed_A_.isZero()) {
                A.topLeftCorner(fixed_A_.rows(), fixed_A_.cols()) += fixed_A_;
                b.head(fixed_b_.size()) += fixed_b_;
            }

            size_t fixed_state_size = fixed_state_vector.getStateSize();
            if (fixed_state_size > 0) {
                Eigen::MatrixXd A00 = A.topLeftCorner(fixed_state_size, fixed_state_size);
                Eigen::MatrixXd A10 = A.bottomLeftCorner(A.rows() - fixed_state_size, fixed_state_size);
                Eigen::MatrixXd A11 = A.bottomRightCorner(A.rows() - fixed_state_size, A.cols() - fixed_state_size);
                Eigen::VectorXd b0 = b.head(fixed_state_size);
                Eigen::VectorXd b1 = b.tail(b.size() - fixed_state_size);

                fixed_A_ = A11 - A10 * A00.llt().solve(A10.transpose());
                fixed_b_ = b1 - A10 * A00.llt().solve(b0);
            } else {
                fixed_A_ = A;
                fixed_b_ = b;
            }

            // Step 6: Remove fixed variables efficiently
            std::unordered_set<slam::eval::StateKey> removal_set(to_remove.begin(), to_remove.end());
            tbb::concurrent_vector<slam::eval::StateKey> new_queue;
            for (const auto& key : variable_queue_) {
                if (!removal_set.count(key)) {
                    new_queue.push_back(key);
                } else {
                    related_var_keys_.erase(key);
                    variables_.erase(key);
                }
            }

            variable_queue_ = std::move(new_queue);
            getStateVector();
        }

        // -----------------------------------------------------------------------------
        // addCostTerm
        // -----------------------------------------------------------------------------

        void SlidingWindowFilter::addCostTerm(slam::problem::costterm::BaseCostTerm::ConstPtr cost_term) {
            // Step 1: Safe insertion into cost_terms_
            cost_terms_.push_back(cost_term);

            // Step 2: Retrieve related variable keys
            KeySet related_keys;
            cost_term->getRelatedVarKeys(related_keys);

            // Step 3: Ensure all related keys exist and update them safely
            for (const auto& key : related_keys) {
                RelatedVarKeysMap::accessor accessor;
                
                // If the key is missing, it's a logic error (it should have been added earlier)
                if (!related_var_keys_.find(accessor, key)) {
                    throw std::runtime_error("[SlidingWindowFilter::addCostTerm] Key " + std::to_string(key) + 
                                            " does not exist in related_var_keys_. Marginalization cannot proceed.");
                }

                // Step 4: Update the key's dependency set safely
                accessor->second.insert(related_keys.begin(), related_keys.end());
            }
        }

        // -----------------------------------------------------------------------------
        // cost
        // -----------------------------------------------------------------------------

        double SlidingWindowFilter::cost() const noexcept {
            return tbb::parallel_reduce(
                tbb::blocked_range<size_t>(0, cost_terms_.size()), 
                0.0,  // Initial cost value
                [&](const tbb::blocked_range<size_t>& range, double local_cost) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        double cost_i = cost_terms_[i]->cost();
                        if (!std::isnan(cost_i)) {
                            local_cost += cost_i;
                        }
                    }
                    return local_cost;
                },
                [](double a, double b) {
                    return a + b;
                }
            );
        }

        // -----------------------------------------------------------------------------
        // getNumberOfCostTerms
        // -----------------------------------------------------------------------------

        unsigned int SlidingWindowFilter::getNumberOfCostTerms() const noexcept {
            return cost_terms_.size();
        }

        // -----------------------------------------------------------------------------
        // getNumberOfVariables
        // -----------------------------------------------------------------------------

        unsigned int SlidingWindowFilter::getNumberOfVariables() const noexcept {
            return variable_queue_.size();
        }

        // -----------------------------------------------------------------------------
        // getStateVector
        // -----------------------------------------------------------------------------

        StateVector::Ptr SlidingWindowFilter::getStateVector() const {
            // Step 1: Reset existing state vectors instead of reassigning
            marginalize_state_vector_->clear();
            active_state_vector_->clear();
            state_vector_->clear();

            // Step 2: Process variables safely
            bool marginalize = true;

            for (const auto &key : variable_queue_) {
                VariableMap::const_accessor accessor;
                if (!variables_.find(accessor, key)) {
                    throw std::runtime_error("[SlidingWindowFilter::getStateVector] Key not found in variables_.");
                }

                const auto &var = accessor->second;

                if (var.marginalize) {
                    if (!marginalize) {
                        throw std::runtime_error("Marginalized variables must be at the first positions in the queue.");
                    }
                    marginalize_state_vector_->addStateVariable(var.variable);
                } else {
                    marginalize = false;
                    active_state_vector_->addStateVariable(var.variable);
                }
                state_vector_->addStateVariable(var.variable);
            }

            return active_state_vector_;
        }

        // -----------------------------------------------------------------------------
        // buildGaussNewtonTerms
        // ----------------------------------------------------------------------------

        void SlidingWindowFilter::buildGaussNewtonTerms(
            Eigen::SparseMatrix<double> &approximate_hessian,
            Eigen::VectorXd &gradient_vector) const {

            // Step 1: Initialize BlockSparseMatrix and BlockVector
            std::vector<unsigned int> sqSizes = state_vector_->getStateBlockSizes();
            slam::blockmatrix::BlockSparseMatrix A_(sqSizes, true);
            slam::blockmatrix::BlockVector b_(sqSizes);

            // Step 2: Parallel computation of Gauss-Newton terms using TBB
            tbb::parallel_for(tbb::blocked_range<size_t>(0, cost_terms_.size()), [&](const tbb::blocked_range<size_t> &range) {
                for (size_t c = range.begin(); c < range.end(); ++c) {
                    cost_terms_[c]->buildGaussNewtonTerms(*state_vector_, &A_, &b_);
                }
            });

            // Step 3: Convert BlockSparseMatrix to Eigen Matrix efficiently
            Eigen::MatrixXd A_upper = A_.toEigen(false);
            Eigen::MatrixXd A = A_upper.selfadjointView<Eigen::Upper>();
            Eigen::VectorXd b = b_.toEigen();

            // Step 4: Add fixed constraints (if any)
            if (fixed_A_.size() > 0) {
                A.topLeftCorner(fixed_A_.rows(), fixed_A_.cols()).noalias() += fixed_A_;
                b.head(fixed_b_.size()).noalias() += fixed_b_;
            }

            // Step 5: Marginalization of fixed variables
            const auto marginalize_state_size = marginalize_state_vector_->getStateSize();
            if (marginalize_state_size > 0) {
                Eigen::Ref<Eigen::MatrixXd> A00 = A.topLeftCorner(marginalize_state_size, marginalize_state_size);
                Eigen::Ref<Eigen::MatrixXd> A10 = A.bottomLeftCorner(A.rows() - marginalize_state_size, marginalize_state_size);
                Eigen::Ref<Eigen::MatrixXd> A11 = A.bottomRightCorner(A.rows() - marginalize_state_size, A.cols() - marginalize_state_size);
                Eigen::Ref<Eigen::VectorXd> b0 = b.head(marginalize_state_size);
                Eigen::Ref<Eigen::VectorXd> b1 = b.tail(b.size() - marginalize_state_size);

                // Compute inverse only once using LLT decomposition
                Eigen::MatrixXd A00_inv = A00.llt().solve(Eigen::MatrixXd::Identity(A00.rows(), A00.cols()));
                
                // Perform marginalization
                approximate_hessian = (A11 - A10 * A00_inv * A10.transpose()).sparseView();
                gradient_vector = b1 - A10 * A00_inv * b0;
            } else {
                approximate_hessian = A.sparseView();
                gradient_vector = b;
            }
        }
    }  // namespace problem
}  // namespace slam