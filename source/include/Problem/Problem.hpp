#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include "source/include/Problem/CostTerm/BaseCostTerm.hpp"
#include "source/include/Problem/StateVector.hpp"

namespace slam {
    namespace problem {

        // -----------------------------------------------------------------------------
        /**
         * @class Problem
         * @brief Interface for a SLAM optimization problem.
         *
         * This class provides:
         * - **State Variable Management:** Adding and retrieving variables.
         * - **Cost Function Handling:** Managing cost terms.
         * - **Optimization Interface:** Constructing Gauss-Newton system for solvers.
         */
        class Problem {
            public:
                using Ptr = std::shared_ptr<Problem>;
                using ConstPtr = std::shared_ptr<const Problem>;

                virtual ~Problem() = default;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves the total number of cost terms.
                 * @return Number of cost terms in the problem.
                 */
                [[nodiscard]] virtual unsigned int getNumberOfCostTerms() const noexcept = 0;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Adds a state variable to the problem.
                 * @param state_var Shared pointer to the state variable.
                 */
                virtual void addStateVariable(const slam::eval::StateVariableBase::Ptr& state_var) = 0;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Adds a cost term to the optimization problem.
                 * @param cost_term Unique pointer to the cost term.
                 */
                virtual void addCostTerm(slam::problem::costterm::BaseCostTerm::ConstPtr cost_term) = 0;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes the total cost of the optimization problem.
                 * @return The accumulated cost from all cost terms.
                 */
                [[nodiscard]] virtual double cost() const noexcept = 0;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves a reference to the state vector.
                 * @return Shared pointer to the state vector.
                 */
                [[nodiscard]] virtual slam::problem::StateVector::Ptr getStateVector() const = 0;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs the Gauss-Newton system.
                 *
                 * Populates:
                 * - The **approximate Hessian** (left-hand side).
                 * - The **gradient vector** (right-hand side).
                 *
                 * @param approximate_hessian Eigen sparse matrix for Hessian approximation.
                 * @param gradient_vector Eigen vector for the gradient.
                 */
                virtual void buildGaussNewtonTerms(
                    Eigen::SparseMatrix<double>& approximate_hessian,
                    Eigen::VectorXd& gradient_vector) const = 0;
        };

    }  // namespace problem
}  // namespace slam
