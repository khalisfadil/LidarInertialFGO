#pragma once

#include <vector>
#include <memory>
#include <tbb/parallel_for.h>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include "Problem/Problem.hpp"

namespace slam {
    namespace problem {

        // -----------------------------------------------------------------------------
        /**
         * @class OptimizationProblem
         * @brief Represents a standard optimization problem in SLAM.
         *
         * This class provides:
         * - **State Variable Management**: Storing and retrieving state variables.
         * - **Cost Function Management**: Adding and evaluating cost terms.
         * - **Gauss-Newton System Construction**: Computing Hessians & gradients.
         * - **Multithreading Support**: Parallel evaluation of cost terms.
         */
        class OptimizationProblem : public Problem {
            public:
                using Ptr = std::shared_ptr<OptimizationProblem>;
                using ConstPtr = std::shared_ptr<const OptimizationProblem>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Creates a shared instance of the optimization problem.
                 * @return Shared pointer to the newly created problem.
                 */
                static Ptr MakeShared() {
                    return std::make_shared<OptimizationProblem>();
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for OptimizationProblem.
                 */
                explicit OptimizationProblem();

                // -----------------------------------------------------------------------------
                /** @brief Adds a state variable to the problem. */
                void addStateVariable(const slam::eval::StateVariableBase::Ptr& state_var) override;

                // -----------------------------------------------------------------------------
                /** @brief Adds a cost term to the problem. */
                void addCostTerm(slam::problem::costterm::BaseCostTerm::ConstPtr cost_term) override;

                // -----------------------------------------------------------------------------
                /** @brief Retrieves the total number of cost terms. */
                [[nodiscard]] unsigned int getNumberOfCostTerms() const noexcept override;

                // -----------------------------------------------------------------------------
                /** @brief Computes the total cost of the optimization problem. */
                [[nodiscard]] double cost() const noexcept override;

                // -----------------------------------------------------------------------------
                /** @brief Retrieves a reference to the state vector. */
                [[nodiscard]] StateVector::Ptr getStateVector() const override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs the Gauss-Newton system.
                 * 
                 * Computes:
                 * - The **approximate Hessian** (left-hand side).
                 * - The **gradient vector** (right-hand side).
                 *
                 * @param approximate_hessian Eigen sparse matrix for Hessian approximation.
                 * @param gradient_vector Eigen vector for the gradient.
                 */
                void buildGaussNewtonTerms(Eigen::SparseMatrix<double>& approximate_hessian,
                                            Eigen::VectorXd& gradient_vector) const override;

            private:

                // -----------------------------------------------------------------------------
                /** @brief Collection of cost terms. */
                std::vector<slam::problem::costterm::BaseCostTerm::ConstPtr> cost_terms_;

                // -----------------------------------------------------------------------------
                /** @brief Collection of state variables. */
                std::vector<slam::eval::StateVariableBase::Ptr> state_vars_;

                // -----------------------------------------------------------------------------
                /** @brief State vector (created when calling `getStateVector()`). */
                StateVector::Ptr state_vector_ = StateVector::MakeShared();
        };
    }  // namespace problem
}  // namespace slam
