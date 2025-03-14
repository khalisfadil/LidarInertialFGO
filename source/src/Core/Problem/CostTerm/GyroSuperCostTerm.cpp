#include "Core/Problem/CostTerm/GyroSuperCostTerm.hpp"
#include <iostream>

namespace slam {
    namespace problem {
        namespace costterm {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            GyroSuperCostTerm::Ptr GyroSuperCostTerm::MakeShared(const Interface::ConstPtr &interface,
                                                                    const Time &time1, const Time &time2,
                                                                    const slam::eval::Evaluable<BiasType>::ConstPtr &bias1,
                                                                    const slam::eval::Evaluable<BiasType>::ConstPtr &bias2,
                                                                    const Options &options) {
                return std::make_shared<GyroSuperCostTerm>(interface, time1, time2, bias1, bias2, options);
            }

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------
            
            GyroSuperCostTerm::GyroSuperCostTerm(const Interface::ConstPtr &interface,
                                           const Time &time1, const Time &time2,
                                           const slam::eval::Evaluable<BiasType>::ConstPtr &bias1,
                                           const slam::eval::Evaluable<BiasType>::ConstPtr &bias2,
                                           const Options &options)
                : interface_(interface),
                time1_(time1),
                time2_(time2),
                bias1_(bias1),
                bias2_(bias2),
                options_(options),
                knot1_(interface_->get(time1)),
                knot2_(interface_->get(time2)){

                // Configure loss functions based on options
                auto configureLoss = [](LOSS_FUNC type, double sigma) -> slam::problem::lossfunc::BaseLossFunc::Ptr {
                    switch (type) {
                        case LOSS_FUNC::L2: return slam::problem::lossfunc::L2LossFunc::MakeShared();
                        case LOSS_FUNC::DCS: return slam::problem::lossfunc::DcsLossFunc::MakeShared(sigma);
                        case LOSS_FUNC::CAUCHY: return slam::problem::lossfunc::CauchyLossFunc::MakeShared(sigma);
                        case LOSS_FUNC::GM: return slam::problem::lossfunc::GemanMcClureLossFunc::MakeShared(sigma);
                        default: return nullptr;
                    }
                };

                gyro_loss_func_ = configureLoss(options_.gyro_loss_func, options_.gyro_loss_sigma);

                // Initialize Jacobians
                jac_vel_.block<3, 3>(0, 3).setIdentity();
                jac_bias_gyro_.block<3, 3>(0, 3).setIdentity() *= -1;

                // Apply SE(2) constraints if enabled
                if (options_.se2) {
                    jac_vel_.block<1, 1>(0, 5).setConstant(1);  // Equivalent to `jac_vel_(0, 5) = 1`
                    jac_bias_gyro_.block<1, 1>(0, 5).setConstant(-1);  // Equivalent to `jac_bias_(0, 5) = -1`
                }

                // Configure noise models
                gyro_noise_model_ = slam::problem::noisemodel::StaticNoiseModel<3>::MakeShared(options_.r_imu_ang.asDiagonal());
            }

            // -----------------------------------------------------------------------------
            // cost
            // -----------------------------------------------------------------------------

            double GyroSuperCostTerm::cost() const noexcept {
                using namespace slam::eval::se3;
                using namespace slam::eval::vspace;

                // Forward propagate variables
                const auto T1_ = knot1_->getPose()->forward();
                const auto w1_ = knot1_->getVelocity()->forward();
                const auto T2_ = knot2_->getPose()->forward();
                const auto w2_ = knot2_->getVelocity()->forward();
                const auto b1_ = bias1_->forward();
                const auto b2_ = bias2_->forward();

                // Extract values
                const auto& T1 = T1_->value();
                const auto& w1 = w1_->value();
                const auto& T2 = T2_->value();
                const auto& w2 = w2_->value();
                const auto& b1 = b1_->value();
                const auto& b2 = b2_->value();

                // Compute transformation and Jacobians
                const auto xi_21 = (T2 / T1).vec();
                const liemath::se3::Transformation T_21(xi_21, 0);
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);
                const auto J_21_inv_w2 = J_21_inv * w2;

                double total_cost = 0.0;

                // Use TBB Parallel Reduction
                total_cost = tbb::parallel_reduce(
                    tbb::blocked_range<size_t>(0, imu_data_vec_.size()), 0.0,
                    [&](const tbb::blocked_range<size_t>& range, double local_cost) -> double {
                        for (size_t i = range.begin(); i < range.end(); ++i) {
                            const auto& imu_data = imu_data_vec_[i];
                            const double ts = imu_data.timestamp;

                            // Declare omega and lambda before lookup
                            Eigen::Matrix4d omega, lambda;

                            // Use TBB concurrent_hash_map accessor for thread-safe lookup
                            tbb::concurrent_hash_map<double, std::pair<Eigen::Matrix4d, Eigen::Matrix4d>>::const_accessor accessor;
                            if (interp_mats_.find(accessor, ts)) {
                                omega = accessor->second.first;
                                lambda = accessor->second.second;
                            } else {
                                // Skip computation if data is missing
                                continue;
                            }

                            // Compute interpolated pose, velocity, and acceleration
                            const Eigen::Matrix<double, 6, 1> xi_i1 =
                                lambda(0, 1) * w1 +
                                omega(0, 0) * xi_21 +
                                omega(0, 1) * J_21_inv_w2;

                            const Eigen::Matrix<double, 6, 1> xi_j1 =
                                lambda(1, 1) * w1 +
                                omega(1, 0) * xi_21 +
                                omega(1, 1) * J_21_inv_w2;

                            // Compute interpolated velocity & acceleration
                            const Eigen::Matrix<double, 6, 1> w_i = liemath::se3::vec2jac(xi_i1) * xi_j1;

                            // Interpolated bias
                            const double ratio = (ts - knot1_->getTime().seconds()) /
                                                (knot2_->getTime().seconds() - knot1_->getTime().seconds());
                            const Eigen::Matrix<double, 6, 1> bias_i = (1 - ratio) * b1 + ratio * b2;

                            // Compute errors
                            Eigen::Matrix<double, 3, 1> raw_error_gyro;
                            if (options_.se2) {
                                raw_error_gyro.setZero();  // Ensure other components remain unchanged
                                raw_error_gyro(2, 0) = imu_data.ang_vel(2, 0) + w_i(5, 0) - bias_i(5, 0);
                            } else {
                                raw_error_gyro = imu_data.ang_vel + w_i.block<3, 1>(3, 0) - bias_i.block<3, 1>(3, 0);
                            }

                            local_cost += gyro_loss_func_->cost(gyro_noise_model_->getWhitenedErrorNorm(raw_error_gyro));
                        }
                        return local_cost;
                    },
                    std::plus<>());

                return total_cost;
            }

            // -----------------------------------------------------------------------------
            // getRelatedVarKeys
            // -----------------------------------------------------------------------------

            void GyroSuperCostTerm::buildGaussNewtonTerms(const StateVector &state_vec,
                                                        slam::blockmatrix::BlockSparseMatrix *approximate_hessian,
                                                        slam::blockmatrix::BlockVector *gradient_vector) const {
                using namespace slam::eval::se3;
                using namespace slam::eval::vspace;

                // Extract Forward Values (Pose, Velocity, Bias)
                const auto T1_ = knot1_->getPose()->forward();
                const auto w1_ = knot1_->getVelocity()->forward();
                const auto T2_ = knot2_->getPose()->forward();
                const auto w2_ = knot2_->getVelocity()->forward();
                const auto b1_ = bias1_->forward();
                const auto b2_ = bias2_->forward();

                const auto T1 = T1_->value(); // 4x4 SE(3) pose
                const auto w1 = w1_->value(); // 6D velocity
                const auto T2 = T2_->value();
                const auto w2 = w2_->value();
                const auto b1 = b1_->value(); // 6D bias
                const auto b2 = b2_->value();

                // Compute Relative Transformation Between Knots
                const auto xi_21 = (T2 / T1).vec();
                const liemath::se3::Transformation T_21(xi_21, 0);
                const auto Ad_T_21 = liemath::se3::tranAd(T_21.matrix());
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);
                const auto w2_j_21_inv = 0.5 * liemath::se3::curlyhat(w2) * J_21_inv;
                const auto J_21_inv_w2 = J_21_inv * w2;

                // Define type aliases for clarity and consistency
                using Matrix36x36 = Eigen::Matrix<double, 36, 36>;
                using Matrix36x1 = Eigen::Matrix<double, 36, 1>;
                using PairType = std::pair<Matrix36x36, Matrix36x1>;

                // Parallel Computation Using TBB
                auto [A, b] = tbb::parallel_reduce(
                    tbb::blocked_range<int>(0, imu_data_vec_.size()),
                    PairType(Matrix36x36::Zero(), Matrix36x1::Zero()), // Concrete initial value
                    [&](const tbb::blocked_range<int> &range, PairType A_b) -> PairType {
                        Matrix36x36 &A_local = A_b.first;
                        Matrix36x1 &b_local = A_b.second;

                        for (int i = range.begin(); i < range.end(); ++i) {
                            const double ts = imu_data_vec_[i].timestamp;
                            const IMUData &imu_data = imu_data_vec_[i];

                            // Retrieve Precomputed Interpolation Matrices
                            tbb::concurrent_hash_map<double, std::pair<Eigen::Matrix4d, Eigen::Matrix4d>>::const_accessor accessor;
                            if (!interp_mats_.find(accessor, ts)) {
                                throw std::runtime_error("Timestamp not found in interpolation matrices.");
                            }
                            const auto &omega = accessor->second.first;
                            const auto &lambda = accessor->second.second;

                            // Optimized Velocity
                            const Eigen::Matrix<double, 6, 1> xi_i1 = lambda(0, 1) * w1 + omega(0, 0) * xi_21 + omega(0, 1) * J_21_inv_w2;
                            const Eigen::Matrix<double, 6, 1> xi_j1 = lambda(1, 1) * w1 + omega(1, 0) * xi_21 + omega(1, 1) * J_21_inv_w2 + w2_j_21_inv * w2;
                            const Eigen::Matrix<double, 6, 1> w_i = liemath::se3::vec2jac(xi_i1) * xi_j1;

                            // Interpolated Bias Computation
                            Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();
                            const double T_inv = 1.0 / (knot2_->getTime().seconds() - knot1_->getTime().seconds());
                            const double omega_ = (ts - knot1_->getTime().seconds()) * T_inv;
                            const double lambda_ = 1.0 - omega_;
                            Eigen::Matrix<double, 6, 1> bias_i = lambda_ * b1 + omega_ * b2;

                            Eigen::Matrix<double, 6, 12> interp_jac_bias;
                            interp_jac_bias << lambda_ * I6, omega_ * I6;

                            // Velocity Interpolation Jacobians
                            Eigen::Matrix<double, 6, 24> interp_jac_vel;
                            interp_jac_vel.setZero();

                            const Eigen::Matrix<double, 6, 6> J_i1 = liemath::se3::vec2jac(xi_i1);
                            const Eigen::Matrix<double, 6, 6> xi_j1_ch = -0.5 * liemath::se3::curlyhat(xi_j1);
                            const Eigen::Matrix<double, 6, 6> omega_J_21_inv = J_i1 * (omega(1, 0) * J_21_inv + omega(1, 1) * w2_j_21_inv);
                            const Eigen::Matrix<double, 6, 6> xi_j1_ch_omega = xi_j1_ch * (omega(0, 0) * J_21_inv + omega(0, 1) * w2_j_21_inv);

                            Eigen::Matrix<double, 6, 6> w = omega_J_21_inv + xi_j1_ch_omega;
                            interp_jac_vel.block<6, 6>(0, 0) = -w * Ad_T_21; // T1
                            interp_jac_vel.block<6, 6>(0, 6) = (lambda(1, 1) * J_i1 + lambda(0, 1) * xi_j1_ch); // w1
                            interp_jac_vel.block<6, 6>(0, 12) = w; // T2
                            interp_jac_vel.block<6, 6>(0, 18) = omega(1, 1) * J_i1 * J_21_inv + omega(0, 1) * xi_j1_ch * J_21_inv; // w2

                            // Evaluate, Weight, and Whiten Gyroscope Error
                            Eigen::Matrix<double, 3, 1> raw_error_gyro = imu_data.ang_vel;
                            if (options_.se2) {
                                raw_error_gyro.setZero();
                                raw_error_gyro(2, 0) = imu_data.ang_vel(2, 0) + w_i(5, 0) - bias_i(5, 0);
                            } else {
                                raw_error_gyro.noalias() += w_i.block<3, 1>(3, 0) - bias_i.block<3, 1>(3, 0);
                            }

                            const Eigen::Matrix<double, 3, 1> white_error_gyro = gyro_noise_model_->whitenError(raw_error_gyro);
                            const double sqrt_w_gyro = std::sqrt(gyro_loss_func_->weight(white_error_gyro.squaredNorm()));
                            const Eigen::Matrix<double, 3, 1> error_gyro = sqrt_w_gyro * white_error_gyro;

                            // Gyro Measurement Jacobians
                            Eigen::Matrix<double, 3, 36> G;
                            G.setZero();
                            G.block<3, 24>(0, 0).noalias() = sqrt_w_gyro * gyro_noise_model_->getSqrtInformation() * jac_vel_ * interp_jac_vel;
                            G.block<3, 12>(0, 24).noalias() = sqrt_w_gyro * gyro_noise_model_->getSqrtInformation() * jac_bias_gyro_ * interp_jac_bias;

                            // Update Local Hessian and Gradient
                            A_local.noalias() += G.transpose() * G;
                            b_local.noalias() -= G.transpose() * error_gyro;
                        }
                        return A_b;
                    },
                    [](const PairType &a, const PairType &b) -> PairType {
                        return PairType(a.first + b.first, a.second + b.second); // Force evaluation to concrete type
                    });

                // Optimized Active State Extraction
                constexpr int num_states = 6;
                std::array<bool, num_states> active = {
                    knot1_->getPose()->active(),
                    knot1_->getVelocity()->active(),
                    knot2_->getPose()->active(),
                    knot2_->getVelocity()->active(),
                    bias1_->active(),
                    bias2_->active()
                };

                // Preallocate memory for keys
                std::vector<slam::eval::StateKey> keys(num_states, -1);

                // Parallel Jacobian Computation Using TBB with Proper Type Casting
                tbb::parallel_for(0, num_states, [&](int i) {
                    if (!active[i]) return;

                    slam::eval::StateKeyJacobians jacs;
                    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero(); 

                    switch (i) {
                        case 0: {
                            auto T1node = std::static_pointer_cast<slam::eval::Node<PoseType>>(T1_);
                            knot1_->getPose()->backward(lhs, T1node, jacs);
                            break;
                        }
                        case 1: {
                            auto w1node = std::static_pointer_cast<slam::eval::Node<VelType>>(w1_);
                            knot1_->getVelocity()->backward(lhs, w1node, jacs);
                            break;
                        }
                        case 2: {
                            auto T2node = std::static_pointer_cast<slam::eval::Node<PoseType>>(T2_);
                            knot2_->getPose()->backward(lhs, T2node, jacs);
                            break;
                        }
                        case 3: {
                            auto w2node = std::static_pointer_cast<slam::eval::Node<VelType>>(w2_);
                            knot2_->getVelocity()->backward(lhs, w2node, jacs);
                            break;
                        }
                        case 4: {
                            auto b1node = std::static_pointer_cast<slam::eval::Node<BiasType>>(b1_);
                            bias1_->backward(lhs, b1node, jacs);
                            break;
                        }
                        case 5: {
                            auto b2node = std::static_pointer_cast<slam::eval::Node<BiasType>>(b2_);
                            bias2_->backward(lhs, b2node, jacs);
                            break;
                        }
                    }

                    const auto jacmap = jacs.get();
                    assert(jacmap.size() == 1);
                    keys[i] = jacmap.begin()->first;
                });

                // Parallel Hessian and Gradient Update Using TBB
                tbb::parallel_for(0, num_states, [&](int i) {
                    if (!active[i]) return;

                    unsigned int blkIdx1 = state_vec.getStateBlockIndex(keys[i]);
                    Eigen::MatrixXd newGradTerm = b.block<6, 1>(i * 6, 0);
                    gradient_vector->mapAt(blkIdx1) += newGradTerm;

                    tbb::parallel_for(i, num_states, [&](int j) {
                        if (!active[j]) return;

                        unsigned int blkIdx2 = state_vec.getStateBlockIndex(keys[j]);
                        unsigned int row = std::min(blkIdx1, blkIdx2);
                        unsigned int col = std::max(blkIdx1, blkIdx2);

                        const Eigen::MatrixXd newHessianTerm = (blkIdx1 <= blkIdx2)
                            ? A.block<6, 6>(i * 6, j * 6)
                            : A.block<6, 6>(j * 6, i * 6);

                        approximate_hessian->add(row, col, newHessianTerm);
                    });
                });
            }
        }  // namespace costterm
    }  // namespace problem
}  // namespace slam