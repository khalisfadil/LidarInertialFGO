#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>
#include "LGMath/LieGroupMath.hpp"
#include "Core/slam.hpp"

using namespace slam;

/**
 * \brief Example that loads and solves simple bundle adjustment problems.
 * It optimizes a B-spline trajectory based on velocity measurements and saves
 * the pre-optimization and post-optimization velocities to a file.
 */
int main(int argc, char** argv) {
    const double T = 1.0;
    traj::Time knot_spacing(0.4);

    // Create a trajectory interface
    traj::bspline::Interface traj(knot_spacing);

    // Define a set of velocity measurements (time, velocity)
    std::vector<std::pair<traj::Time, Eigen::Matrix<double, 6, 1>>> w_iv_inv_meas;
    w_iv_inv_meas.emplace_back(traj::Time(0.1 * T), 0.0 * Eigen::Matrix<double, 6, 1>::Ones());
    w_iv_inv_meas.emplace_back(traj::Time(0.2 * T), 0.2 * Eigen::Matrix<double, 6, 1>::Ones());
    w_iv_inv_meas.emplace_back(traj::Time(0.3 * T), 0.4 * Eigen::Matrix<double, 6, 1>::Ones());
    w_iv_inv_meas.emplace_back(traj::Time(0.4 * T), 0.6 * Eigen::Matrix<double, 6, 1>::Ones());
    w_iv_inv_meas.emplace_back(traj::Time(0.5 * T), 0.4 * Eigen::Matrix<double, 6, 1>::Ones());
    w_iv_inv_meas.emplace_back(traj::Time(0.6 * T), 0.2 * Eigen::Matrix<double, 6, 1>::Ones());
    w_iv_inv_meas.emplace_back(traj::Time(0.7 * T), 0.0 * Eigen::Matrix<double, 6, 1>::Ones());
    w_iv_inv_meas.emplace_back(traj::Time(0.8 * T), 0.2 * Eigen::Matrix<double, 6, 1>::Ones());
    w_iv_inv_meas.emplace_back(traj::Time(0.9 * T), 0.4 * Eigen::Matrix<double, 6, 1>::Ones());

    std::vector<problem::costterm::BaseCostTerm::Ptr> cost_terms;
    const auto loss_func = problem::lossfunc::L2LossFunc::MakeShared();
    const auto noise_model = problem::noisemodel::StaticNoiseModel<6>::MakeShared(Eigen::Matrix<double, 6, 6>::Identity());

    // Create cost terms based on the velocity measurements
    for (auto& meas : w_iv_inv_meas) {
        const auto error_func = eval::vspace::vspace_error<6>(traj.getVelocityInterpolator(meas.first), meas.second);
        cost_terms.emplace_back(problem::costterm::WeightedLeastSqCostTerm<6>::MakeShared(error_func, noise_model, loss_func));
    }

    // Initialize the optimization problem
    problem::OptimizationProblem problem;

    // Add state variables (B-spline trajectory parameters)
    traj.addStateVariables(problem);

    // Add measurement cost terms
    for (const auto& cost : cost_terms) problem.addCostTerm(cost);

    // Save the pre-optimization velocities and post-optimization velocities to a file
    std::ofstream outFile("trajectory_results.txt");
    outFile << "Time Measured_Vel Pre_Optimize_Vel Estimated_Vel\n";

    // Create the Gauss-Newton solver and solve the optimization problem
    solver::GaussNewtonSolver::Params params;
    params.verbose = true;
    solver::GaussNewtonSolver solver(problem, params);

    // Save pre-optimization velocities before the solver optimization
    for (const auto& meas : w_iv_inv_meas) {
        // Pre-optimization velocity estimation (using the unoptimized trajectory)
        Eigen::Matrix<double, 6, 1> pre_optimized_velocity = traj.getVelocityInterpolator(meas.first)->value();

        // Save measured and pre-optimization velocities
        outFile << meas.first.seconds() << " "
                << meas.second.transpose() << " "
                << pre_optimized_velocity.transpose() << " ";
    }

    // Optimize the trajectory
    solver.optimize();

    // Save the post-optimization velocities (after the solver optimization)
    for (const auto& meas : w_iv_inv_meas) {
        // Post-optimization velocity estimation (using the optimized trajectory)
        Eigen::Matrix<double, 6, 1> estimated_velocity = traj.getVelocityInterpolator(meas.first)->value();

        // Write the optimized velocity to the file
        outFile << estimated_velocity.transpose() << "\n";
    }

    outFile.close();

    return 0;
}
