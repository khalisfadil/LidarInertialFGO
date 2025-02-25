#pragma once

#include <deque>
#include <unordered_map>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <tbb/parallel_for.h>

#include "source/include/Problem/Problem.hpp"
#include "source/include/Problem/StateVector.hpp"
#include "source/include/Problem/CostTerm/BaseCostTerm.hpp"

namespace slam {
    namespace problem {

        // -----------------------------------------------------------------------------
        /**
         * @class SlidingWindowFilter
         * @brief Implements a **Sliding Window Filter** for SLAM optimization.
         *
         * **Key Features:**
         * - **State Management:** Allows adding, tracking, and marginalizing state variables.
         * - **Cost Function Handling:** Stores and processes cost terms affecting the system.
         * - **Efficient Marginalization:** Automatically removes old variables while preserving dependencies.
         * - **Gauss-Newton System Construction:** Computes Hessians & gradients for optimization.
         * - **Multithreading Support:** Uses **Intel TBB** for parallel cost term evaluation.
         *
         * This filter maintains a **sliding window** of state variables, retaining only the most
         * recent ones while marginalizing older ones based on dependencies.
         */
        class SlidingWindowFilter : public Problem {
            public:
                using Ptr = std::shared_ptr<SlidingWindowFilter>;
                using KeySet = slam::problem::costterm::BaseCostTerm::KeySet;  ///< KeySet for tracking variable dependencies

                // -----------------------------------------------------------------------------
                /**
                 * @brief Creates a shared instance of the Sliding Window Filter.
                 * @return Shared pointer to the created instance.
                 */
                static Ptr MakeShared();

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for Sliding Window Filter.
                 */
                explicit SlidingWindowFilter();

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves the total number of cost terms.
                 * @return Number of cost terms in the current window.
                 */
                [[nodiscard]] unsigned int getNumberOfCostTerms() const noexcept override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves the total number of state variables in the window.
                 * @return Number of active state variables in the filter.
                 */
                [[nodiscard]] unsigned int getNumberOfVariables() const noexcept;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes the total cost of the optimization problem.
                 * @return The accumulated cost from all cost terms.
                 */
                [[nodiscard]] double cost() const noexcept override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves the active state vector (excluding marginalized variables).
                 * @return Shared pointer to the active state vector.
                 */
                [[nodiscard]] StateVector::Ptr getStateVector() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs the Gauss-Newton system.
                 *
                 * This function computes:
                 * - The **approximate Hessian** (left-hand side).
                 * - The **gradient vector** (right-hand side).
                 *
                 * @param approximate_hessian Eigen sparse matrix for Hessian approximation.
                 * @param gradient_vector Eigen vector for the gradient.
                 */
                void buildGaussNewtonTerms(Eigen::SparseMatrix<double>& approximate_hessian,
                                           Eigen::VectorXd& gradient_vector) const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Adds a state variable to the sliding window.
                 * @param variable Shared pointer to the state variable.
                 */
                void addStateVariable(const slam::eval::StateVariableBase::Ptr& variable) override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Adds multiple state variables to the sliding window.
                 * @param variables Vector of shared pointers to state variables.
                 */
                void addStateVariable(const std::vector<slam::eval::StateVariableBase::Ptr>& variables);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Marks a state variable for marginalization.
                 * @param variable Shared pointer to the state variable to be marginalized.
                 */
                void marginalizeVariable(const slam::eval::StateVariableBase::Ptr& variable);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Marks multiple state variables for marginalization.
                 * @param variables Vector of shared pointers to state variables.
                 */
                void marginalizeVariable(const std::vector<slam::eval::StateVariableBase::Ptr>& variables);

                // -----------------------------------------------------------------------------
                /**
                 * @brief Adds a cost term to the sliding window filter.
                 * @param cost_term Shared pointer to the cost term to be added.
                 */
                void addCostTerm(slam::problem::costterm::BaseCostTerm::ConstPtr cost_term) override;

            private:
                // -----------------------------------------------------------------------------
                /**
                 * @struct Variable
                 * @brief Represents a state variable in the filter.
                 *
                 * Each variable has:
                 * - A **pointer** to the state variable itself.
                 * - A **flag** indicating whether the variable should be marginalized.
                 */
                struct Variable {
                    explicit Variable(const slam::eval::StateVariableBase::Ptr& v, bool m)
                        : variable(v), marginalize(m) {}
                    slam::eval::StateVariableBase::Ptr variable = nullptr;  ///< Pointer to the state variable.
                    bool marginalize = false;  ///< Flag indicating if the variable is marked for marginalization.
                };

                // -----------------------------------------------------------------------------
                /** @brief Maps state keys to their corresponding variables. */
                using VariableMap = std::unordered_map<slam::eval::StateKey, Variable, slam::eval::StateKeyHash>;
                VariableMap variables_;

                // -----------------------------------------------------------------------------
                /** @brief Maintains an ordered queue of state variables for the sliding window. */
                std::deque<slam::eval::StateKey> variable_queue_;

                // -----------------------------------------------------------------------------
                /** @brief Tracks variable dependencies for marginalization. */
                std::unordered_map<slam::eval::StateKey, KeySet, slam::eval::StateKeyHash> related_var_keys_;

                // -----------------------------------------------------------------------------
                /** @brief Collection of cost terms affecting the current optimization window. */
                std::vector<slam::problem::costterm::BaseCostTerm::ConstPtr> cost_terms_;

                // -----------------------------------------------------------------------------
                /** @brief Fixed linearized system from marginalized variables (stored as dense for now). */
                Eigen::MatrixXd fixed_A_;
                Eigen::VectorXd fixed_b_;

                // -----------------------------------------------------------------------------
                /** @brief Active state vector (only contains non-marginalized variables). */
                StateVector::Ptr active_state_vector_ = std::make_shared<StateVector>();

                // -----------------------------------------------------------------------------
                /** @brief State vector containing marginalized variables. */
                StateVector::Ptr marginalize_state_vector_ = std::make_shared<StateVector>();

                // -----------------------------------------------------------------------------
                /** @brief Complete state vector containing both active and marginalized variables. */
                StateVector::Ptr state_vector_ = std::make_shared<StateVector>();
        };

    }  // namespace problem
}  // namespace slam
