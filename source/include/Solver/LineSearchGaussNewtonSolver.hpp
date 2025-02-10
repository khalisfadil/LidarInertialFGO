#pragma once

#include "source/include/Solver/GaussNewtonSolver.hpp"

namespace slam {
    namespace solver {

        // ----------------------------------------------------------------------------
        /**
         * @class LineSearchGaussNewtonSolver
         * @brief Gauss-Newton solver with backtracking line search.
         */
        class LineSearchGaussNewtonSolver : public GaussNewtonSolver {
            public:
                // ----------------------------------------------------------------------------
                struct Params : public GaussNewtonSolver::Params {
                    /// Step size reduction factor during backtracking
                    double backtrack_multiplier = 0.5;
                    /// Maximum number of backtracking steps before giving up
                    unsigned int max_backtrack_steps = 10;
                };

                // ----------------------------------------------------------------------------
                explicit LineSearchGaussNewtonSolver(slam::problem::Problem& problem, const Params& params);

            private:
                // ----------------------------------------------------------------------------
                bool linearizeSolveAndUpdate(double& cost, double& grad_norm) override;

                const Params& params_; ///< Store reference to avoid unnecessary copies
        };

    }  // namespace solver
}  // namespace slam
