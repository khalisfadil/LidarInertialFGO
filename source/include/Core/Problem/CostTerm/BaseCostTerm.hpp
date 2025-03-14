#pragma once

#include <memory>
#include <tbb/concurrent_unordered_set.h>  // ✅ Use unordered set, not hash map

#include "Core/MatrixOperator/BlockSparseMatrix.hpp"
#include "Core/MatrixOperator/BlockVector.hpp"
#include "Core/Problem/StateVector.hpp"
#include "Core/Evaluable/StateKey.hpp"

namespace slam {
    namespace problem {
        namespace costterm {

            // -----------------------------------------------------------------------------
            /**
             * @class BaseCostTerm
             * @brief Interface for a cost function term contributing to the objective function.
             *
             * **Responsibilities:**
             * - **Cost Evaluation:** Computes the scalar cost term.
             * - **Variable Dependency Tracking:** Identifies relevant state variables.
             * - **Optimization Contribution:** Builds Gauss-Newton terms (Hessian & gradient).
             */
            class BaseCostTerm {
            public:
                using Ptr = std::shared_ptr<BaseCostTerm>; ///< Shared pointer to allow cost term sharing
                using ConstPtr = std::shared_ptr<const BaseCostTerm>;

                /** @brief Explicit virtual destructor for safe inheritance */
                virtual ~BaseCostTerm() = default;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Computes the cost contribution to the objective function.
                 * @return Scalar cost value.
                 */
                [[nodiscard]] virtual double cost() const noexcept = 0;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Retrieves variable keys that this cost term depends on.
                 * @param keys Set of related variable keys.
                 */
                using KeySet = tbb::concurrent_unordered_set<slam::eval::StateKey, slam::eval::StateKeyHasher, slam::eval::StateKeyEqual>;
                virtual void getRelatedVarKeys(KeySet &keys) const noexcept = 0;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Adds contributions to the Gauss-Newton system.
                 *
                 * Computes:
                 * - The **Hessian approximation** (left-hand side).
                 * - The **gradient vector** (right-hand side).
                 *
                 * @param state_vec Current state vector.
                 * @param approximate_hessian Pointer to the Hessian matrix.
                 * @param gradient_vector Pointer to the gradient vector.
                 */
                virtual void buildGaussNewtonTerms(
                    const StateVector &state_vec,
                    slam::blockmatrix::BlockSparseMatrix *approximate_hessian,
                    slam::blockmatrix::BlockVector *gradient_vector) const = 0;
            };

        }  // namespace costterm
    }  // namespace problem
}  // namespace slam
