#pragma once

// blkmat
#include "source/include/MatrixOperator/BlockMatrix.hpp"
#include "source/include/MatrixOperator/BlockSparseMatrix.hpp"
#include "source/include/MatrixOperator/BlockVector.hpp"

// common
#include "source/include/MatrixOperator/BlockVector.hpp"

// evaluables (including variable)
#include "source/include/Evaluable/Evaluable.hpp"
#include "source/include/Evaluable/StateVariable.hpp"

#include "source/include/Evaluable/imu/Evaluables.hpp"
#include "source/include/Evaluable/point2point/Evaluables.hpp"
#include "source/include/Evaluable/se3/Evaluables.hpp"
#include "source/include/Evaluable/vspace/Evaluables.hpp"

// problem
#include "source/include/Problem/CostTerm/WeightLeastSqCostTerm.hpp"
#include "source/include/Problem/LossFunc/LossFunc.hpp"
#include "source/include/Problem/NoiseModel/StaticNoiseModel.hpp"
#include "source/include/Problem/NoiseModel/DynamicNoiseModel.hpp"
#include "source/include/Problem/OptimizationProblem.hpp"
#include "source/include/Problem/SlidingWindowFilter.hpp"

// solver
#include "source/include/Solver/Covariance.hpp"
#include "source/include/Solver/DoglegGaussNewtonSolver.hpp"
#include "source/include/Solver/GaussNewtonSolver.hpp"
#include "source/include/Solver/LevMarqGaussNewtonSolver.hpp"
#include "source/include/Solver/LineSearchGaussNewtonSolver.hpp"

// trajectory
#include "source/include/Trajectory/Bspline/Interface.hpp"
#include "source/include/Trajectory/ConstAcceleration/Interface.hpp"
#include "source/include/Trajectory/ConstVelocity/Interface.hpp"
#include "source/include/Trajectory/Singer/Interface.hpp"
