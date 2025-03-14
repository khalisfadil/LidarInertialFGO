#pragma once

// blkmat
#include "Core/MatrixOperator/BlockMatrix.hpp"
#include "Core/MatrixOperator/BlockSparseMatrix.hpp"
#include "Core/MatrixOperator/BlockVector.hpp"

// common
#include "Core/Common/Timer.hpp"

// evaluables (including variable)
#include "Core/Evaluable/Evaluable.hpp"
#include "Core/Evaluable/StateVariable.hpp"

#include "Core/Evaluable/imu/Evaluables.hpp"
#include "Core/Evaluable/point2point/Evaluables.hpp"
#include "Core/Evaluable/se3/Evaluables.hpp"
#include "Core/Evaluable/vspace/Evaluables.hpp"

// problem
#include "Core/Problem/CostTerm/WeightLeastSqCostTerm.hpp"
#include "Core/Problem/LossFunc/LossFunc.hpp"
#include "Core/Problem/NoiseModel/StaticNoiseModel.hpp"
#include "Core/Problem/NoiseModel/DynamicNoiseModel.hpp"
#include "Core/Problem/OptimizationProblem.hpp"
#include "Core/Problem/SlidingWindowFilter.hpp"

// solver
#include "Core/Solver/Covariance.hpp"
#include "Core/Solver/DoglegGaussNewtonSolver.hpp"
#include "Core/Solver/GaussNewtonSolver.hpp"
#include "Core/Solver/LevMarqGaussNewtonSolver.hpp"
#include "Core/Solver/LineSearchGaussNewtonSolver.hpp"

// trajectory
#include "Core/Trajectory/Bspline/Interface.hpp"
#include "Core/Trajectory/ConstAcceleration/Interface.hpp"
#include "Core/Trajectory/ConstVelocity/Interface.hpp"
#include "Core/Trajectory/Singer/Interface.hpp"
