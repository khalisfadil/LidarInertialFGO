cmake_minimum_required(VERSION 3.16)
project(slam)

option(BUILD_TESTING "Build tests for slam" OFF)

# Compiler setup
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable parallel LTO and optimizations
add_compile_options(-march=native -O3 -Wall -pedantic -Wno-unused-function -flto)
add_link_options(-flto=$(nproc))  # Parallel LTO with all cores

# CMake policies (set before finding packages to suppress Open3D warning)
cmake_policy(SET CMP0072 NEW)

# Find required dependencies
set(Open3D_DIR "/usr/local/lib/cmake/Open3D/") # Ensure correct directory
set(TBB_DIR "/usr/lib/x86_64-linux-gnu/cmake/TBB") # Ensure correct directory
find_package(Open3D REQUIRED)
find_package(Boost 1.83 REQUIRED COMPONENTS system)
find_package(TBB REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
# find_package(PCL 1.14 REQUIRED)
# find_package(OpenMP REQUIRED)

# Define include directories for reuse
set(INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/source/include)

# Core library
add_library(core STATIC
    # source/src/Core/Evaluable/imu/AccErrorEvaluator.cpp
    # source/src/Core/Evaluable/imu/DmiErrorEvaluator.cpp
    # source/src/Core/Evaluable/imu/GyroErrorEvaluator.cpp
    # source/src/Core/Evaluable/imu/GyroErrorEvaluatorSE2.cpp
    # source/src/Core/Evaluable/imu/IMUErrorEvaluator.cpp

    # source/src/Core/Evaluable/point2point/p2pErrorEvaluator.cpp
    # source/src/Core/Evaluable/point2point/p2planeErrorEvaluator.cpp
    # source/src/Core/Evaluable/point2point/p2planeGlobalPerturbEvaluator.cpp
    # source/src/Core/Evaluable/point2point/VelErrorEvaluator.cpp
    # source/src/Core/Evaluable/point2point/YawErrorEvaluator.cpp
    # source/src/Core/Evaluable/point2point/YawVelErrorEvaluator.cpp

    # source/src/Core/Evaluable/se3/ComposeEvaluator.cpp
    # source/src/Core/Evaluable/se3/ComposeInverseEvaluator.cpp
    # source/src/Core/Evaluable/se3/ComposeVelocityEvaluator.cpp
    # source/src/Core/Evaluable/se3/ExpMapEvaluator.cpp
    # source/src/Core/Evaluable/se3/InverseEvaluator.cpp
    # source/src/Core/Evaluable/se3/LogMapEvaluator.cpp
    # source/src/Core/Evaluable/se3/PoseInterpolator.cpp
    # source/src/Core/Evaluable/se3/Se3ErrorEvaluator.cpp
    # source/src/Core/Evaluable/se3/Se3ErrorGlobalPerturbEvaluator.cpp
    # source/src/Core/Evaluable/se3/Se3StateVariable.cpp
    # source/src/Core/Evaluable/se3/Se3StateVariableGlobalPerturb.cpp

    # source/src/LGMath/r3/Operations.cpp
    # source/src/LGMath/se3/Operations.cpp
    # source/src/LGMath/se3/Transformation.cpp
    # source/src/LGMath/se3/TransformationWithCovariance.cpp
    # source/src/LGMath/so3/Operations.cpp
    # source/src/LGMath/so3/Rotations.cpp
    # source/src/LGMath/CommonMath.cpp

    # source/src/Core/MatrixOperator/BlockDimensionIndexing.cpp
    # source/src/Core/MatrixOperator/BlockMatrix.cpp
    # source/src/Core/MatrixOperator/BlockMatrixBase.cpp
    # source/src/Core/MatrixOperator/BlockMatrixIndexing.cpp
    # source/src/Core/MatrixOperator/BlockSparseMatrix.cpp
    # source/src/Core/MatrixOperator/BlockVector.cpp

    # source/src/Core/Problem/CostTerm/GyroSuperCostTerm.cpp
    # source/src/Core/Problem/CostTerm/IMUSuperCostTerm.cpp
    # source/src/Core/Problem/OptimizationProblem.cpp
    # source/src/Core/Problem/SlidingWindowFilter.cpp
    # source/src/Core/Problem/StateVector.cpp

    # source/src/Core/Solver/Covariance.cpp
    # source/src/Core/Solver/DoglegGaussNewtonSolver.cpp
    # source/src/Core/Solver/GaussNewtonSolver.cpp
    # source/src/Core/Solver/GaussNewtonSolverNVA.cpp
    # source/src/Core/Solver/LevMarqGaussNewtonSolver.cpp
    # source/src/Core/Solver/LineSearchGaussNewtonSolver.cpp
    # source/src/Core/Solver/SolverBase.cpp

    # source/src/Core/Trajectory/Bspline/Interface.cpp
    # source/src/Core/Trajectory/Bspline/VelocityInterpolator.cpp

    # source/src/Core/Trajectory/ConstAcceleration/Evaluable/composeCurlyhatEvaluator.cpp
    # source/src/Core/Trajectory/ConstAcceleration/AccelerationExtrapolator.cpp
    # source/src/Core/Trajectory/ConstAcceleration/AccelerationInterpolator.cpp
    # source/src/Core/Trajectory/ConstAcceleration/Interface.cpp
    # source/src/Core/Trajectory/ConstAcceleration/PoseExtrapolator.cpp
    # source/src/Core/Trajectory/ConstAcceleration/PoseInterpolator.cpp
    # source/src/Core/Trajectory/ConstAcceleration/PriorFactor.cpp
    # source/src/Core/Trajectory/ConstAcceleration/Variables.cpp
    # source/src/Core/Trajectory/ConstAcceleration/VelocityExtrapolator.cpp
    # source/src/Core/Trajectory/ConstAcceleration/VelocityInterpolator.cpp

    # source/src/Core/Trajectory/ConstVelocity/Evaluable/JinvVelocityEvaluator.cpp
    # source/src/Core/Trajectory/ConstVelocity/Evaluable/JVelocityEvaluator.cpp
    # source/src/Core/Trajectory/ConstVelocity/Interface.cpp
    # source/src/Core/Trajectory/ConstVelocity/PoseExtrapolator.cpp
    # source/src/Core/Trajectory/ConstVelocity/PoseInterpolator.cpp
    # source/src/Core/Trajectory/ConstVelocity/PriorFactor.cpp
    # source/src/Core/Trajectory/ConstVelocity/Variables.cpp
    # source/src/Core/Trajectory/ConstVelocity/VelocityInterpolator.cpp

    # source/src/Core/Trajectory/Singer/AccelerationExtrapolator.cpp
    # source/src/Core/Trajectory/Singer/AccelerationInterpolator.cpp
    # source/src/Core/Trajectory/Singer/Interface.cpp
    # source/src/Core/Trajectory/Singer/PoseExtrapolator.cpp
    # source/src/Core/Trajectory/Singer/PoseInterpolator.cpp
    # source/src/Core/Trajectory/Singer/PriorFactor.cpp
    # source/src/Core/Trajectory/Singer/VelocityExtrapolator.cpp
    # source/src/Core/Trajectory/Singer/VelocityInterpolator.cpp
    # source/src/Core/Trajectory/Time.cpp

    source/src/DymCluster/ClusterExtraction.cpp
    source/src/DymCluster/Hungarian.cpp
    source/src/DymMap/OccupancyMap.cpp
    source/src/DymMap/Pipeline.cpp
    source/src/Utils/Open3d/voxelGrid.cpp
)

# Include directories for the core library
target_include_directories(core PUBLIC ${INCLUDE_DIRECTORIES})

# Link libraries to the core library
target_link_libraries(core
  PUBLIC
    TBB::tbb
    Boost::system
    Open3D::Open3D
    Eigen3::Eigen
    # ${PCL_LIBRARIES}
    # OpenMP::OpenMP_CXX
)

# Install the headers and library
install(DIRECTORY source/include/ DESTINATION include)
install(TARGETS core DESTINATION lib)

# 🔥 **Fix for `BSplineTrajectoryExample` missing headers**
add_executable(DymMapAndDymClusterExample source/samples/DymMapAndDymClusterExample.cpp)
target_include_directories(DymMapAndDymClusterExample PRIVATE ${CMAKE_SOURCE_DIR}/source/include)
target_link_libraries(DymMapAndDymClusterExample PRIVATE core)

