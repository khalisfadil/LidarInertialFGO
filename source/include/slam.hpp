#pragma once

// blkmat
#include "MatrixOperator/BlockMatrix.hpp"
#include "MatrixOperator/BlockSparseMatrix.hpp"
#include "MatrixOperator/BlockVector.hpp"

// common
#include "Common/Timer.hpp"

// evaluables (including variable)
#include "Evaluable/Evaluable.hpp"
#include "Evaluable/StateVariable.hpp"

#include "Evaluable/imu/Evaluables.hpp"
#include "Evaluable/point2point/Evaluables.hpp"
#include "Evaluable/se3/Evaluables.hpp"
#include "Evaluable/vspace/Evaluables.hpp"

// problem
#include "Problem/CostTerm/WeightLeastSqCostTerm.hpp"
#include "Problem/LossFunc/LossFunc.hpp"
#include "Problem/NoiseModel/StaticNoiseModel.hpp"
#include "Problem/NoiseModel/DynamicNoiseModel.hpp"
#include "Problem/OptimizationProblem.hpp"
#include "Problem/SlidingWindowFilter.hpp"

// solver
#include "Solver/Covariance.hpp"
#include "Solver/DoglegGaussNewtonSolver.hpp"
#include "Solver/GaussNewtonSolver.hpp"
#include "Solver/LevMarqGaussNewtonSolver.hpp"
#include "Solver/LineSearchGaussNewtonSolver.hpp"

// trajectory
#include "Trajectory/Bspline/Interface.hpp"
#include "Trajectory/ConstAcceleration/Interface.hpp"
#include "Trajectory/ConstVelocity/Interface.hpp"
#include "Trajectory/Singer/Interface.hpp"
