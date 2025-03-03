cmake_minimum_required(VERSION 3.16)
project(slam)

option(BUILD_TESTING "Build tests for slam" OFF)

# Compiler setup (assumed to be GNU)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-march=native -O3 -Wall -pedantic -Wno-unused-function)

# CMake policies
cmake_policy(SET CMP0072 NEW) # OpenGL handling

# Find required dependencies
set(Open3D_DIR "/usr/local/lib/cmake/Open3D/") # Ensure correct directory
set(TBB_DIR "/usr/lib/x86_64-linux-gnu/cmake/TBB") # Ensure correct directory
find_package(Open3D REQUIRED)
find_package(Boost 1.83 REQUIRED COMPONENTS system)
find_package(TBB REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(PCL 1.14 REQUIRED)
find_package(OpenMP REQUIRED)

# Define include directories for reuse
set(INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/source/include/Common

    ${CMAKE_SOURCE_DIR}/source/include/Evaluable
    ${CMAKE_SOURCE_DIR}/source/include/Evaluable/imu
    ${CMAKE_SOURCE_DIR}/source/include/Evaluable/point2point
    ${CMAKE_SOURCE_DIR}/source/include/Evaluable/se3
    ${CMAKE_SOURCE_DIR}/source/include/Evaluable/vspace

    ${CMAKE_SOURCE_DIR}/source/include/LGMath
    ${CMAKE_SOURCE_DIR}/source/include/MatrixOperator
    ${CMAKE_SOURCE_DIR}/source/include/Problem
    ${CMAKE_SOURCE_DIR}/source/include/Solver
    ${CMAKE_SOURCE_DIR}/source/include/Trajectory
)

# Core library
add_library(core STATIC
    source/src/Evaluable/imu/AccErrorEvaluator.cpp
    source/src/Evaluable/imu/DmiErrorEvaluator.cpp
    source/src/Evaluable/imu/GyroErrorEvaluator.cpp
    source/src/Evaluable/imu/GyroErrorEvaluatorSE2.cpp
    source/src/Evaluable/imu/IMUErrorEvaluator.cpp

    source/src/Evaluable/point2point/p2pErrorEvaluator.cpp
    source/src/Evaluable/point2point/p2planeErrorEvaluator.cpp
    source/src/Evaluable/point2point/p2planeGlobalPerturbEvaluator.cpp
    source/src/Evaluable/point2point/VelErrorEvaluator.cpp
    source/src/Evaluable/point2point/YawErrorEvaluator.cpp
    source/src/Evaluable/point2point/YawVelErrorEvaluator.cpp

    source/src/Evaluable/se3/ComposeEvaluator.cpp
    source/src/Evaluable/se3/ComposeInverseEvaluator.cpp
    source/src/Evaluable/se3/ComposeVelocityEvaluator.cpp
    source/src/Evaluable/se3/ExpMapEvaluator.cpp
    source/src/Evaluable/se3/InverseEvaluator.cpp
    source/src/Evaluable/se3/LogMapEvaluator.cpp
    source/src/Evaluable/se3/PoseInterpolator.cpp
    source/src/Evaluable/se3/Se3ErrorEvaluator.cpp
    source/src/Evaluable/se3/Se3ErrorGlobalPerturbEvaluator.cpp
    source/src/Evaluable/se3/Se3StateVariable.cpp
    source/src/Evaluable/se3/Se3StateVariableGlobalPerturb.cpp

    source/src/Evaluable/vspace/AdditionEvaluator.cpp
    source/src/Evaluable/vspace/MatrixMultEvaluator.cpp
    source/src/Evaluable/vspace/MergeEvaluator.cpp
    source/src/Evaluable/vspace/NegationEvaluator.cpp
    source/src/Evaluable/vspace/PreIntVelocityStateVar.cpp
    source/src/Evaluable/vspace/ScalarMultEvaluator.cpp
    source/src/Evaluable/vspace/VSpaceErrorEvaluator.cpp
    source/src/Evaluable/vspace/VSpaceInterpolator.cpp
    source/src/Evaluable/vspace/VSpaceStateVar.cpp

    source/src/LGMath/r3/Operations.cpp
    source/src/LGMath/se3/Operations.cpp
    source/src/LGMath/se3/Transformation.cpp
    source/src/LGMath/se3/TransformationWithCovariance.cpp
    source/src/LGMath/so3/Operations.cpp
    source/src/LGMath/so3/Rotations.cpp
    source/src/LGMath/CommonMath.cpp

    source/src/MatrixOperator/BlockDimensionIndexing.cpp
    source/src/MatrixOperator/BlockMatrix.cpp
    source/src/MatrixOperator/BlockMatrixBase.cpp
    source/src/MatrixOperator/BlockMatrixIndexing.cpp
    source/src/MatrixOperator/BlockSparseMatrix.cpp
    source/src/MatrixOperator/BlockVector.cpp

    source/src/Problem/CostTerm/GyroSuperCostTerm.cpp
    source/src/Problem/CostTerm/IMUSuperCostTerm.cpp
    source/src/Problem/CostTerm/WeightLeastSqCostTerm.cpp
    source/src/Problem/OptimizationProblem.cpp
    source/src/Problem/SlidingWindowFilter.cpp
    source/src/Problem/StateVector.cpp

    source/src/Solver/Covariance.cpp
    source/src/Solver/DoglegGaussNewtonSolver.cpp
    source/src/Solver/GaussNewtonSolver.cpp
    source/src/Solver/GaussNewtonSolverNVA.cpp
    source/src/Solver/LevMarqGaussNewtonSolver.cpp
    source/src/Solver/LineSearchGaussNewtonSolver.cpp
    source/src/Solver/SolverBase.cpp

    source/src/Trajectory/Bspline/Interface.cpp
    source/src/Trajectory/Bspline/VelocityInterpolator.cpp

    source/src/Trajectory/ConstAcceleration/Evaluable/composeCurlyhatEvaluator.cpp
    source/src/Trajectory/ConstAcceleration/AccelerationExtrapolator.cpp
    source/src/Trajectory/ConstAcceleration/AccelerationInterpolator.cpp
    source/src/Trajectory/ConstAcceleration/Interface.cpp
    source/src/Trajectory/ConstAcceleration/PoseExtrapolator.cpp
    source/src/Trajectory/ConstAcceleration/PoseInterpolator.cpp
    source/src/Trajectory/ConstAcceleration/PriorFactor.cpp
    source/src/Trajectory/ConstAcceleration/Variables.cpp
    source/src/Trajectory/ConstAcceleration/VelocityExtrapolator.cpp
    source/src/Trajectory/ConstAcceleration/VelocityInterpolator.cpp

    source/src/Trajectory/ConstVelocity/Evaluable/JinvVelocityEvaluator.cpp
    source/src/Trajectory/ConstVelocity/Evaluable/JVelocityEvaluator.cpp
    source/src/Trajectory/ConstVelocity/Interface.cpp
    source/src/Trajectory/ConstVelocity/PoseExtrapolator.cpp
    source/src/Trajectory/ConstVelocity/PoseInterpolator.cpp
    source/src/Trajectory/ConstVelocity/PriorFactor.cpp
    source/src/Trajectory/ConstVelocity/Variables.cpp
    source/src/Trajectory/ConstVelocity/VelocityInterpolator.cpp

    source/src/Trajectory/Singer/AccelerationExtrapolator.cpp
    source/src/Trajectory/Singer/AccelerationInterpolator.cpp
    source/src/Trajectory/Singer/Interface.cpp
    source/src/Trajectory/Singer/PoseExtrapolator.cpp
    source/src/Trajectory/Singer/PoseInterpolator.cpp
    source/src/Trajectory/Singer/PriorFactor.cpp
    source/src/Trajectory/Singer/VelocityExtrapolator.cpp
    source/src/Trajectory/Singer/VelocityInterpolator.cpp
    source/src/Trajectory/Time.cpp
)

# Include directories for the core library
target_include_directories(core PUBLIC ${INCLUDE_DIRECTORIES})

# Link libraries to the core library
target_link_libraries(core
  PRIVATE
    TBB::tbb
    Boost::system
    Open3D::Open3D
    Eigen3::Eigen
    ${PCL_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Install the headers and library
install(
  DIRECTORY source/include/
  DESTINATION include
)

install(
  TARGETS core
  DESTINATION lib
)

# Executables
add_executable(BSplineTrajectoryExample source/samples/BSplineTrajectoryExample.cpp)

# Optionally uncomment and add more executables as needed
# add_executable(LandmarkSLAMExample samples/LandmarkSLAMExample.cpp)
# add_executable(RadialVelMeasWithConstVelTraj samples/RadialVelMeasWithConstVelTraj.cpp)
# add_executable(RadialVelocityMeasurement samples/RadialVelocityMeasurement.cpp)
# add_executable(SimplePoseGraphRelax samples/SimplePoseGraphRelax.cpp)
# add_executable(SpherePoseGraphRelax samples/SpherePoseGraphRelax.cpp)
# add_executable(SimplePointCloudAlignment samples/SimplePointCloudAlignment.cpp)
# add_executable(TrustRegionExample samples/TrustRegionExample.cpp)
# add_executable(SimpleConstVelTrajPrior samples/SimpleConstVelTrajPrior.cpp)
# add_executable(SimpleBundleAdjustment samples/SimpleBundleAdjustment.cpp)
# add_executable(SimpleBundleAdjustmentRelLand samples/SimpleBundleAdjustmentRelLand.cpp)
# add_executable(SimpleBundleAdjustmentFullRel samples/SimpleBundleAdjustmentFullRel.cpp)
# add_executable(SimpleBAandConstVelTrajPrior samples/SimpleBAandConstVelTrajPrior.cpp)
# add_executable(SlidingWindowFilterExample samples/SlidingWindowFilterExample.cpp)
# add_executable(MotionPriors samples/MotionPriors.cpp)
