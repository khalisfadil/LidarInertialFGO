#include "Problem/CostTerm/IMUSuperCostTerm.hpp"
#include <iostream>

namespace slam {
    namespace problem {
        namespace costterm {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            IMUSuperCostTerm::Ptr IMUSuperCostTerm::MakeShared(const Interface::ConstPtr &interface, const Time time1, const Time time2,
                                                                const slam::eval::Evaluable<BiasType>::ConstPtr &bias1,
                                                                const slam::eval::Evaluable<BiasType>::ConstPtr &bias2,
                                                                const slam::eval::Evaluable<PoseType>::ConstPtr &transform_i_to_m_1,
                                                                const slam::eval::Evaluable<PoseType>::ConstPtr &transform_i_to_m_2,
                                                                const Options &options) {
                return std::make_shared<IMUSuperCostTerm>(interface, time1, time2, bias1,
                                                            bias2, transform_i_to_m_1,
                                                            transform_i_to_m_2, options);
            }


            // -----------------------------------------------------------------------------
            // IMUSuperCostTerm
            // -----------------------------------------------------------------------------

            IMUSuperCostTerm::IMUSuperCostTerm(const Interface::ConstPtr& interface, 
                                   const Time time1, const Time time2,
                                   const slam::eval::Evaluable<BiasType>::ConstPtr& bias1,
                                   const slam::eval::Evaluable<BiasType>::ConstPtr& bias2,
                                   const slam::eval::Evaluable<PoseType>::ConstPtr& transform_i_to_m_1,
                                   const slam::eval::Evaluable<PoseType>::ConstPtr& transform_i_to_m_2,
                                   const Options& options)
                : interface_(interface),
                time1_(time1),
                time2_(time2),
                bias1_(bias1),
                bias2_(bias2),
                transform_i_to_m_1_(transform_i_to_m_1),
                transform_i_to_m_2_(transform_i_to_m_2),
                options_(options),
                knot1_(interface_->get(time1)),
                knot2_(interface_->get(time2)),
                Qinv_T_(interface_->getQinvPublic((knot2_->getTime() - knot1_->getTime()).seconds(), Eigen::Matrix<double, 6, 1>::Ones())),
                Tran_T_(interface_->getTranPublic((knot2_->getTime() - knot1_->getTime()).seconds())) {

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

                acc_loss_func_ = configureLoss(options_.acc_loss_func, options_.acc_loss_sigma);
                gyro_loss_func_ = configureLoss(options_.gyro_loss_func, options_.gyro_loss_sigma);

                // Initialize Jacobians
                jac_vel_.block<3, 3>(0, 3).setIdentity();
                jac_accel_.block<3, 3>(0, 0).setIdentity();
                jac_bias_accel_.block<3, 3>(0, 0).setIdentity() *= -1;
                jac_bias_gyro_.block<3, 3>(0, 3).setIdentity() *= -1;

                // Apply SE(2) constraints if enabled
                if (options_.se2) {
                    jac_vel_.block<2, 2>(1, 4).setZero();
                    jac_accel_(2, 2) = jac_bias_accel_(2, 2) = 0.;
                    jac_bias_gyro_.block<2, 2>(1, 4).setZero();
                }

                // Configure noise models
                acc_noise_model_ = slam::problem::noisemodel::StaticNoiseModel<3>::MakeShared(options_.r_imu_acc.asDiagonal());
                gyro_noise_model_ = slam::problem::noisemodel::StaticNoiseModel<3>::MakeShared(options_.r_imu_ang.asDiagonal());
            }

            // -----------------------------------------------------------------------------
            // cost
            // -----------------------------------------------------------------------------

            double IMUSuperCostTerm::cost() const noexcept {
                using namespace slam::eval::se3;
                using namespace slam::eval::vspace;

                // Forward propagate variables
                const auto T1_ = knot1_->getPose()->forward();
                const auto w1_ = knot1_->getVelocity()->forward();
                const auto dw1_ = knot1_->getAcceleration()->forward();
                const auto T2_ = knot2_->getPose()->forward();
                const auto w2_ = knot2_->getVelocity()->forward();
                const auto dw2_ = knot2_->getAcceleration()->forward();
                const auto b1_ = bias1_->forward();
                const auto b2_ = bias2_->forward();
                const auto T_mi_1_ = transform_i_to_m_1_->forward();
                const auto T_mi_2_ = transform_i_to_m_2_->forward();

                // Extract values
                const auto& T1 = T1_->value();
                const auto& w1 = w1_->value();
                const auto& dw1 = dw1_->value();
                const auto& T2 = T2_->value();
                const auto& w2 = w2_->value();
                const auto& dw2 = dw2_->value();
                const auto& b1 = b1_->value();
                const auto& b2 = b2_->value();
                const auto& T_mi_1 = T_mi_1_->value();
                const auto& T_mi_2 = T_mi_2_->value();

                // Compute transformation and Jacobians
                const auto xi_21 = (T2 / T1).vec();
                const liemath::se3::Transformation T_21(xi_21, 0);
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);
                const auto J_21_inv_w2 = J_21_inv * w2;
                const auto J_21_inv_curl_dw2 =
                    (-0.5 * liemath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

                double total_cost = 0.0;

                // Use TBB Parallel Reduction
                total_cost = tbb::parallel_reduce(
                    tbb::blocked_range<size_t>(0, imu_data_vec_.size()), 0.0,
                    [&](const tbb::blocked_range<size_t>& range, double local_cost) -> double {
                        for (size_t i = range.begin(); i < range.end(); ++i) {
                            const auto& imu_data = imu_data_vec_[i];
                            const double ts = imu_data.timestamp;

                            // Declare omega and lambda before lookup
                            Eigen::Matrix3d omega, lambda;

                            // Use TBB concurrent_hash_map accessor for thread-safe lookup
                            tbb::concurrent_hash_map<double, std::pair<Eigen::Matrix3d, Eigen::Matrix3d>>::const_accessor accessor;
                            if (interp_mats_.find(accessor, ts)) {
                                omega = accessor->second.first;
                                lambda = accessor->second.second;
                            } else {
                                // Skip computation if data is missing
                                continue;
                            }

                            // Compute interpolated pose, velocity, and acceleration
                            const Eigen::Matrix<double, 6, 1> xi_i1 =
                                lambda(0, 1) * w1 + lambda(0, 2) * dw1 +
                                omega(0, 0) * xi_21 +
                                omega(0, 1) * J_21_inv_w2 +
                                omega(0, 2) * J_21_inv_curl_dw2;

                            const Eigen::Matrix<double, 6, 1> xi_j1 =
                                lambda(1, 1) * w1 + lambda(1, 2) * dw1 +
                                omega(1, 0) * xi_21 +
                                omega(1, 1) * J_21_inv_w2 +
                                omega(1, 2) * J_21_inv_curl_dw2;

                            const Eigen::Matrix<double, 6, 1> xi_k1 =
                                lambda(2, 1) * w1 + lambda(2, 2) * dw1 +
                                omega(2, 0) * xi_21 +
                                omega(2, 1) * J_21_inv_w2 +
                                omega(2, 2) * J_21_inv_curl_dw2;

                            // Compute interpolated transformations
                            const liemath::se3::Transformation T_i1(xi_i1, 0);
                            const liemath::se3::Transformation T_i0 = T_i1 * T1;

                            // Compute interpolated velocity & acceleration
                            const Eigen::Matrix<double, 6, 1> w_i = liemath::se3::vec2jac(xi_i1) * xi_j1;
                            const Eigen::Matrix<double, 6, 1> dw_i =
                                liemath::se3::vec2jac(xi_i1) *
                                (xi_k1 + 0.5 * liemath::se3::curlyhat(xi_j1) * w_i);

                            // Interpolated bias
                            const double ratio = (ts - knot1_->getTime().seconds()) /
                                                (knot2_->getTime().seconds() - knot1_->getTime().seconds());
                            const Eigen::Matrix<double, 6, 1> bias_i = (1 - ratio) * b1 + ratio * b2;

                            // Interpolated T_mi
                            liemath::se3::Transformation transform_i_to_m = T_mi_1;
                            if (transform_i_to_m_1_->active() || transform_i_to_m_2_->active()) {
                                const Eigen::Matrix<double, 6, 1> xi_i1_ =
                                    ratio * (T_mi_2 / T_mi_1).vec();
                                transform_i_to_m = liemath::se3::Transformation(xi_i1_, 0) * T_mi_1;
                            }

                            // Compute errors
                            const Eigen::Matrix3d& C_vm = T_i0.matrix().block<3, 3>(0, 0);
                            const Eigen::Matrix3d& C_mi = transform_i_to_m.matrix().block<3, 3>(0, 0);

                            Eigen::Matrix<double, 3, 1> raw_error_acc =
                                imu_data.lin_acc + dw_i.block<3, 1>(0, 0) +
                                C_vm * C_mi * options_.gravity - bias_i.block<3, 1>(0, 0);

                            if (options_.use_accel)
                                local_cost += acc_loss_func_->cost(acc_noise_model_->getWhitenedErrorNorm(raw_error_acc));

                            Eigen::Matrix<double, 3, 1> raw_error_gyro =
                                imu_data.ang_vel + w_i.block<3, 1>(3, 0) - bias_i.block<3, 1>(3, 0);

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
            
            void IMUSuperCostTerm::getRelatedVarKeys(KeySet &keys) const noexcept {
                knot1_->getPose()->getRelatedVarKeys(keys);
                knot1_->getVelocity()->getRelatedVarKeys(keys);
                knot1_->getAcceleration()->getRelatedVarKeys(keys);
                knot2_->getPose()->getRelatedVarKeys(keys);
                knot2_->getVelocity()->getRelatedVarKeys(keys);
                knot2_->getAcceleration()->getRelatedVarKeys(keys);
                bias1_->getRelatedVarKeys(keys);
                bias2_->getRelatedVarKeys(keys);
                transform_i_to_m_1_->getRelatedVarKeys(keys);
                transform_i_to_m_2_->getRelatedVarKeys(keys);
            }

            // -----------------------------------------------------------------------------
            // initialize_interp_matrices_
            // -----------------------------------------------------------------------------

            void IMUSuperCostTerm::initialize_interp_matrices_() {
                static const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
                using MapType = tbb::concurrent_hash_map<double, std::pair<Eigen::Matrix3d, Eigen::Matrix3d>>;

                // Pre-size the map to reduce rehashing (optional optimization)
                interp_mats_.rehash(imu_data_vec_.size());

                tbb::parallel_for(tbb::blocked_range<size_t>(0, imu_data_vec_.size()),
                                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        const double time = imu_data_vec_[i].timestamp;

                        // Thread-safe lookup or insertion
                        MapType::accessor accessor;
                        if (interp_mats_.find(accessor, time)) {
                            continue; // Skip if already computed
                        }

                        // Compute time deltas
                        const double tau = time - time1_.seconds();
                        const double kappa = knot2_->getTime().seconds() - time;

                        // Precompute required matrices
                        const Matrix18d Q_tau = interface_->getQPublic(tau, ones);
                        const Matrix18d Tran_kappa_T = interface_->getTranPublic(kappa).transpose();
                        const Matrix18d Tran_tau = interface_->getTranPublic(tau);

                        // Compute Omega and Lambda
                        const Matrix18d omega18 = Q_tau * Tran_kappa_T * Qinv_T_;
                        const Matrix18d lambda18 = Tran_tau - omega18 * Tran_T_;

                        // Extract 2x2 blocks into 4x4 matrices, initialized to zero
                        Eigen::Matrix3d omega = Eigen::Matrix3d::Zero();
                        Eigen::Matrix3d lambda = Eigen::Matrix3d::Zero();
                        omega << omega18(0, 0), omega18(0, 6), omega18(0, 12),
                                omega18(6, 0), omega18(6, 6), omega18(6, 12),
                                omega18(12, 0), omega18(12, 6), omega18(12, 12);
                        lambda << lambda18(0, 0), lambda18(0, 6), lambda18(0, 12),
                                lambda18(6, 0), lambda18(6, 6), lambda18(6, 12),
                                lambda18(12, 0), lambda18(12, 6), lambda18(12, 12);

                        // Insert into hash map
                        interp_mats_.insert(accessor, {time, {omega, lambda}});
                    }
                });
            }

            // -----------------------------------------------------------------------------
            // buildGaussNewtonTerms
            // -----------------------------------------------------------------------------

            void IMUSuperCostTerm::buildGaussNewtonTerms(
                const StateVector &state_vec,
                slam::blockmatrix::BlockSparseMatrix *approximate_hessian,
                slam::blockmatrix::BlockVector *gradient_vector) const {

                using namespace slam::eval::se3;
                using namespace slam::eval::vspace;

                // Extract Forward Values (Pose, Velocity, Acceleration, Bias, Transformations)
                const auto T1_ = knot1_->getPose()->forward();
                const auto w1_ = knot1_->getVelocity()->forward();
                const auto dw1_ = knot1_->getAcceleration()->forward();
                const auto T2_ = knot2_->getPose()->forward();
                const auto w2_ = knot2_->getVelocity()->forward();
                const auto dw2_ = knot2_->getAcceleration()->forward();
                const auto b1_ = bias1_->forward();
                const auto b2_ = bias2_->forward();
                const auto T_mi_1_ = transform_i_to_m_1_->forward();
                const auto T_mi_2_ = transform_i_to_m_2_->forward();

                const auto T1 = T1_->value();
                const auto w1 = w1_->value();
                const auto dw1 = dw1_->value();
                const auto T2 = T2_->value();
                const auto w2 = w2_->value();
                const auto dw2 = dw2_->value();
                const auto b1 = b1_->value();
                const auto b2 = b2_->value();
                const auto T_mi_1 = T_mi_1_->value();
                const auto T_mi_2 = T_mi_2_->value();

                // Compute Relative Transformation Between Knots
                const auto xi_21 = (T2 / T1).vec();
                const liemath::se3::Transformation T_21(xi_21, 0);
                const Eigen::Matrix<double, 6, 6> J_21_inv = liemath::se3::vec2jacinv(xi_21);
                const auto J_21_inv_w2 = J_21_inv * w2;
                const auto J_21_inv_curl_dw2 = (-0.5 * liemath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

                // Define type aliases for consistency
                using Matrix60x60 = Eigen::Matrix<double, 60, 60>;
                using Matrix60x1 = Eigen::Matrix<double, 60, 1>;
                using PairType = std::pair<Matrix60x60, Matrix60x1>;

                // Parallel Computation Using TBB
                auto result = tbb::parallel_reduce(
                    tbb::blocked_range<int>(0, imu_data_vec_.size()),
                    PairType(Matrix60x60::Zero(), Matrix60x1::Zero()), // Concrete initial value
                    [&](const tbb::blocked_range<int> &range, PairType A_b) -> PairType {
                        Matrix60x60 &A_local = A_b.first;
                        Matrix60x1 &b_local = A_b.second;

                        for (int i = range.begin(); i < range.end(); ++i) {
                            const double ts = imu_data_vec_[i].timestamp;
                            const IMUData &imu_data = imu_data_vec_[i];

                            // Retrieve Precomputed Interpolation Matrices
                            tbb::concurrent_hash_map<double, std::pair<Eigen::Matrix3d, Eigen::Matrix3d>>::const_accessor accessor;
                            if (!interp_mats_.find(accessor, ts)) {
                                throw std::runtime_error("Timestamp not found in interpolation matrices.");
                            }
                            const auto &omega = accessor->second.first;
                            const auto &lambda = accessor->second.second;

                            // Optimized Pose, Velocity, and Acceleration Interpolation
                            const Eigen::Matrix<double, 6, 1> xi_i1 =
                                lambda(0, 1) * w1 + lambda(0, 2) * dw1 +
                                omega(0, 0) * xi_21 + omega(0, 6) * J_21_inv_w2 + omega(0, 2) * J_21_inv_curl_dw2;

                            const Eigen::Matrix<double, 6, 1> xi_j1 =
                                lambda(1, 1) * w1 + lambda(1, 2) * dw1 +
                                omega(1, 0) * xi_21 + omega(1, 1) * J_21_inv_w2 + omega(1, 2) * J_21_inv_curl_dw2;

                            const Eigen::Matrix<double, 6, 1> xi_k1 =
                                lambda(2, 1) * w1 + lambda(2, 2) * dw1 +
                                omega(2, 0) * xi_21 + omega(2, 1) * J_21_inv_w2 + omega(2, 2) * J_21_inv_curl_dw2;

                            const liemath::se3::Transformation T_i1(xi_i1, 0);
                            const liemath::se3::Transformation T_i0 = T_i1 * T1;

                            const Eigen::Matrix<double, 6, 6> J_i1 = liemath::se3::vec2jac(xi_i1);
                            const Eigen::Matrix<double, 6, 1> w_i = J_i1 * xi_j1;
                            const Eigen::Matrix<double, 6, 1> dw_i = J_i1 * (xi_k1 + 0.5 * liemath::se3::curlyhat(xi_j1) * w_i);

                            // Interpolated Bias Computation
                            Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();
                            const double T_inv = 1.0 / (knot2_->getTime().seconds() - knot1_->getTime().seconds());
                            const double omega_ = (ts - knot1_->getTime().seconds()) * T_inv;
                            const double lambda_ = 1.0 - omega_;

                            Eigen::Matrix<double, 6, 1> bias_i = lambda_ * b1 + omega_ * b2;
                            Eigen::Matrix<double, 6, 12> interp_jac_bias;
                            interp_jac_bias << lambda_ * I6, omega_ * I6;

                            // Interpolated T_mi Computation
                            liemath::se3::Transformation transform_i_to_m = T_mi_1;
                            Eigen::Matrix<double, 6, 12> interp_jac_T_m_i = Eigen::Matrix<double, 6, 12>::Zero();

                            if (transform_i_to_m_1_->active() || transform_i_to_m_2_->active()) {
                                const double alpha_ = (ts - knot1_->getTime().seconds()) * T_inv;
                                const Eigen::Matrix<double, 6, 1> xi_i1_ = alpha_ * (T_mi_2 / T_mi_1).vec();
                                transform_i_to_m = liemath::se3::Transformation(xi_i1_, 0) * T_mi_1;

                                const std::array<double, 4> faulhaber_coeffs_ = {
                                    alpha_,
                                    alpha_ * (alpha_ - 1) / 2.0,
                                    alpha_ * (alpha_ - 1) * (2.0 * alpha_ - 1.0) / 12.0,
                                    alpha_ * alpha_ * (alpha_ - 1) * (alpha_ - 1) / 24.0
                                };

                                const Eigen::Matrix<double, 6, 6> xi_21_curlyhat = liemath::se3::curlyhat((T_mi_2 / T_mi_1).vec());
                                Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
                                Eigen::Matrix<double, 6, 6> xictmp = Eigen::Matrix<double, 6, 6>::Identity();

                                for (const auto &coeff : faulhaber_coeffs_) {
                                    A += coeff * xictmp;
                                    xictmp = xi_21_curlyhat * xictmp;
                                }

                                interp_jac_T_m_i.block<6, 6>(0, 0).noalias() = Eigen::Matrix<double, 6, 6>::Identity() - A;
                                interp_jac_T_m_i.block<6, 6>(0, 6).noalias() = A;
                            }

                            // Interpolation Jacobians
                            Eigen::Matrix<double, 6, 36> interp_jac_pose, interp_jac_vel, interp_jac_acc;
                            interp_jac_pose.setZero();
                            interp_jac_vel.setZero();
                            interp_jac_acc.setZero();

                            const Eigen::Matrix<double, 6, 6> xi_j1_ch = -0.5 * liemath::se3::curlyhat(xi_j1);
                            const Eigen::Matrix<double, 6, 6> curlyhat_xi_j1 = liemath::se3::curlyhat(xi_j1);
                            const Eigen::Matrix<double, 6, 6> curlyhat_w_i = liemath::se3::curlyhat(w_i);
                            const Eigen::Matrix<double, 6, 6> curlyhat_xi_j1_sq = curlyhat_xi_j1 * curlyhat_xi_j1;

                            Eigen::Matrix<double, 6, 6> J_prep_2, J_prep_3;
                            J_prep_2.noalias() = J_i1 * (-0.5 * curlyhat_w_i + 0.5 * curlyhat_xi_j1 * J_i1);
                            J_prep_3.noalias() = -0.25 * J_i1 * curlyhat_xi_j1_sq - 0.5 * liemath::se3::curlyhat(xi_k1 + 0.5 * curlyhat_xi_j1 * w_i);

                            const Eigen::Matrix<double, 6, 6> curlyhat_w2 = liemath::se3::curlyhat(w2);
                            const Eigen::Matrix<double, 6, 6> curlyhat_w2_sq = curlyhat_w2 * curlyhat_w2;
                            const Eigen::Matrix<double, 6, 6> curlyhat_dw2 = liemath::se3::curlyhat(dw2);
                            const Eigen::Matrix<double, 6, 6> J_21_inv_curlyhat_w2 = liemath::se3::curlyhat(J_21_inv * w2);

                            // Pose Jacobian
                            Eigen::Matrix<double, 6, 6> w;
                            w.noalias() = J_i1 * (
                                omega(0, 0) * I6 +
                                omega(0, 1) * 0.5 * curlyhat_w2 +
                                omega(0, 2) * (0.25 * curlyhat_w2_sq + 0.5 * curlyhat_dw2)
                            ) * J_21_inv;

                            interp_jac_pose.block<6, 6>(0, 0).noalias() = -w * T_21.adjoint() + T_i1.adjoint();
                            interp_jac_pose.block<6, 6>(0, 6).noalias() = lambda(0, 1) * J_i1;
                            interp_jac_pose.block<6, 6>(0, 12).noalias() = lambda(0, 2) * J_i1;
                            interp_jac_pose.block<6, 6>(0, 18).noalias() = w;
                            interp_jac_pose.block<6, 6>(0, 24).noalias() = omega(0, 1) * J_i1 * J_21_inv + omega(0, 2) * -0.5 * J_i1 * (J_21_inv_curlyhat_w2 - curlyhat_w2 * J_21_inv);
                            interp_jac_pose.block<6, 6>(0, 30).noalias() = omega(0, 2) * J_i1 * J_21_inv;

                            // Velocity Jacobian
                            const Eigen::Matrix<double, 6, 6> v_omega_mat_1 =
                                omega(1, 0) * I6 +
                                omega(1, 1) * 0.5 * curlyhat_w2 +
                                omega(1, 2) * (0.25 * curlyhat_w2_sq + 0.5 * curlyhat_dw2);

                            const Eigen::Matrix<double, 6, 6> v_omega_mat_0 =
                                omega(0, 0) * I6 +
                                omega(0, 1) * 0.5 * curlyhat_w2 +
                                omega(0, 2) * (0.25 * curlyhat_w2_sq + 0.5 * curlyhat_dw2);

                            w.noalias() = J_i1 * v_omega_mat_1 * J_21_inv + xi_j1_ch * v_omega_mat_0 * J_21_inv;
                            interp_jac_vel.block<6, 6>(0, 0).noalias() = -w * T_21.adjoint();
                            interp_jac_vel.block<6, 6>(0, 6).noalias() = J_i1 * lambda(1, 1) + xi_j1_ch * lambda(0, 1);
                            interp_jac_vel.block<6, 6>(0, 12).noalias() = J_i1 * lambda(1, 2) + xi_j1_ch * lambda(0, 2);
                            interp_jac_vel.block<6, 6>(0, 18).noalias() = w;
                            interp_jac_vel.block<6, 6>(0, 24).noalias() = J_i1 * (omega(1, 1) * J_21_inv - 0.5 * omega(1, 2) * (J_21_inv_curlyhat_w2 - curlyhat_w2 * J_21_inv)) +
                                                                        xi_j1_ch * (omega(0, 1) * J_21_inv - 0.5 * omega(0, 2) * (J_21_inv_curlyhat_w2 - curlyhat_w2 * J_21_inv));
                            interp_jac_vel.block<6, 6>(0, 30).noalias() = J_i1 * (omega(1, 2) * J_21_inv) + xi_j1_ch * (omega(0, 2) * J_21_inv);

                            // Acceleration Jacobian
                            const Eigen::Matrix<double, 6, 6> a_omega_mat_2 =
                                omega(2, 0) * I6 +
                                omega(2, 1) * 0.5 * curlyhat_w2 +
                                omega(2, 2) * (0.25 * curlyhat_w2_sq + 0.5 * curlyhat_dw2);

                            const Eigen::Matrix<double, 6, 6> a_omega_mat_1 =
                                omega(1, 0) * I6 +
                                omega(1, 1) * 0.5 * curlyhat_w2 +
                                omega(1, 2) * (0.25 * curlyhat_w2_sq + 0.5 * curlyhat_dw2);

                            const Eigen::Matrix<double, 6, 6> a_omega_mat_0 =
                                omega(0, 0) * I6 +
                                omega(0, 1) * 0.5 * curlyhat_w2 +
                                omega(0, 2) * (0.25 * curlyhat_w2_sq + 0.5 * curlyhat_dw2);

                            w.noalias() = J_i1 * a_omega_mat_2 * J_21_inv +
                                        J_prep_2 * a_omega_mat_1 * J_21_inv +
                                        J_prep_3 * a_omega_mat_0 * J_21_inv;

                            interp_jac_acc.block<6, 6>(0, 0).noalias() = -w * T_21.adjoint();
                            interp_jac_acc.block<6, 6>(0, 6).noalias() = J_i1 * lambda(2, 1) + J_prep_2 * lambda(1, 1) + J_prep_3 * lambda(0, 1);
                            interp_jac_acc.block<6, 6>(0, 12).noalias() = J_i1 * lambda(2, 2) + J_prep_2 * lambda(1, 2) + J_prep_3 * lambda(0, 2);
                            interp_jac_acc.block<6, 6>(0, 18).noalias() = w;
                            interp_jac_acc.block<6, 6>(0, 24).noalias() = J_i1 * (omega(2, 1) * J_21_inv - 0.5 * omega(2, 2) * (J_21_inv_curlyhat_w2 - curlyhat_w2 * J_21_inv)) +
                                                                        J_prep_2 * (omega(1, 1) * J_21_inv - 0.5 * omega(1, 2) * (J_21_inv_curlyhat_w2 - curlyhat_w2 * J_21_inv)) +
                                                                        J_prep_3 * (omega(0, 1) * J_21_inv - 0.5 * omega(0, 2) * (J_21_inv_curlyhat_w2 - curlyhat_w2 * J_21_inv));
                            interp_jac_acc.block<6, 6>(0, 30).noalias() = J_i1 * (omega(2, 2) * J_21_inv) +
                                                                        J_prep_2 * (omega(1, 2) * J_21_inv) + J_prep_3 * (omega(0, 2) * J_21_inv);

                            // Extract rotation matrices
                            const Eigen::Matrix3d &C_vm = T_i0.matrix().block<3, 3>(0, 0);
                            const Eigen::Matrix3d &C_mi = transform_i_to_m.matrix().block<3, 3>(0, 0);

                            // Measurement Jacobian Computation
                            Eigen::Matrix<double, 3, 1> raw_error_acc;
                            if (options_.se2) {
                                raw_error_acc.head<2>().noalias() = imu_data.lin_acc.head<2>() + dw_i.head<2>() - bias_i.head<2>();
                                raw_error_acc(2) = 0.0;
                            } else {
                                raw_error_acc.noalias() = imu_data.lin_acc + dw_i.head<3>() + C_vm * C_mi * options_.gravity - bias_i.head<3>();
                            }

                            const Eigen::Matrix<double, 3, 1> white_error_acc = acc_noise_model_->whitenError(raw_error_acc);
                            const double sqrt_w_acc = std::sqrt(acc_loss_func_->weight(white_error_acc.squaredNorm()));
                            const Eigen::Matrix<double, 3, 1> error_acc = sqrt_w_acc * white_error_acc;

                            Eigen::Matrix<double, 3, 24> Gmeas;
                            Gmeas.setZero();
                            Gmeas.block<3, 6>(0, 6).noalias() = jac_accel_;
                            Gmeas.block<3, 6>(0, 12).noalias() = jac_bias_accel_;

                            if (options_.se2) {
                                Gmeas.row(2).setZero();
                            } else if (options_.use_accel) {
                                const Eigen::Matrix3d hat_gravity = liemath::so3::hat(C_vm * C_mi * options_.gravity);
                                Gmeas.block<3, 3>(0, 3).noalias() = -hat_gravity;
                                Gmeas.block<3, 3>(0, 21).noalias() = -C_vm * liemath::so3::hat(C_mi * options_.gravity);
                            }

                            Gmeas = (sqrt_w_acc * acc_noise_model_->getSqrtInformation()) * Gmeas;

                            // Measurement Jacobians for Interpolation
                            Eigen::Matrix<double, 6, 60> G;
                            G.setZero();

                            G.block<3, 36>(0, 0).noalias() = Gmeas.block<3, 6>(0, 0) * interp_jac_pose +
                                                            Gmeas.block<3, 6>(0, 6) * interp_jac_acc;
                            G.block<3, 12>(0, 36).noalias() = Gmeas.block<3, 6>(0, 12) * interp_jac_bias;
                            G.block<3, 12>(0, 48).noalias() = Gmeas.block<3, 6>(0, 18) * interp_jac_T_m_i;

                            // Gyro Error Computation
                            Eigen::Matrix<double, 3, 1> raw_error_gyro;
                            if (options_.se2) {
                                raw_error_gyro.setZero();
                                raw_error_gyro(2) = imu_data.ang_vel(2) + w_i(5) - bias_i(5);
                            } else {
                                raw_error_gyro.noalias() = imu_data.ang_vel + w_i.tail<3>() - bias_i.tail<3>();
                            }

                            const Eigen::Matrix<double, 3, 1> white_error_gyro = gyro_noise_model_->whitenError(raw_error_gyro);
                            const double sqrt_w_gyro = std::sqrt(gyro_loss_func_->weight(white_error_gyro.squaredNorm()));
                            const Eigen::Matrix<double, 3, 1> error_gyro = sqrt_w_gyro * white_error_gyro;

                            const Eigen::Matrix<double, 3, 3> gyro_sqrt_info = gyro_noise_model_->getSqrtInformation();
                            G.block<3, 36>(3, 0).noalias() = sqrt_w_gyro * gyro_sqrt_info * jac_vel_ * interp_jac_vel;
                            G.block<3, 12>(3, 36).noalias() = sqrt_w_gyro * gyro_sqrt_info * jac_bias_gyro_ * interp_jac_bias;

                            // Hessian and Gradient Computation
                            Eigen::Matrix<double, 6, 1> error;
                            error << error_acc, error_gyro;

                            A_local.noalias() += G.transpose() * G;
                            b_local.noalias() -= G.transpose() * error;
                        }

                        return A_b;
                    },
                    [](const PairType &a, const PairType &b) -> PairType {
                        return PairType(a.first + b.first, a.second + b.second); // Force concrete type
                    });

                // Use the result to update approximate_hessian and gradient_vector
                Eigen::Matrix<double, 60, 60> A = result.first;
                Eigen::Matrix<double, 60, 1> b = result.second;

                // Active State Extraction
                constexpr int num_states = 10;
                std::array<bool, num_states> active = {
                    knot1_->getPose()->active(),
                    knot1_->getVelocity()->active(),
                    knot1_->getAcceleration()->active(),
                    knot2_->getPose()->active(),
                    knot2_->getVelocity()->active(),
                    knot2_->getAcceleration()->active(),
                    bias1_->active(),
                    bias2_->active(),
                    transform_i_to_m_1_->active(),
                    transform_i_to_m_2_->active()
                };

                std::vector<slam::eval::StateKey> keys(num_states, -1);

                // Parallel Jacobian Computation with Proper Type Casting
                tbb::parallel_for(0, num_states, [&](int i) {
                    if (!active[i]) return;

                    slam::eval::StateKeyJacobians jacs;
                    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero(); // Retained as requested

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
                            auto dw1node = std::static_pointer_cast<slam::eval::Node<AccType>>(dw1_);
                            knot1_->getAcceleration()->backward(lhs, dw1node, jacs);
                            break;
                        }
                        case 3: {
                            auto T2node = std::static_pointer_cast<slam::eval::Node<PoseType>>(T2_);
                            knot2_->getPose()->backward(lhs, T2node, jacs);
                            break;
                        }
                        case 4: {
                            auto w2node = std::static_pointer_cast<slam::eval::Node<VelType>>(w2_);
                            knot2_->getVelocity()->backward(lhs, w2node, jacs);
                            break;
                        }
                        case 5: {
                            auto dw2node = std::static_pointer_cast<slam::eval::Node<AccType>>(dw2_);
                            knot2_->getAcceleration()->backward(lhs, dw2node, jacs);
                            break;
                        }
                        case 6: {
                            auto b1node = std::static_pointer_cast<slam::eval::Node<BiasType>>(b1_);
                            bias1_->backward(lhs, b1node, jacs);
                            break;
                        }
                        case 7: {
                            auto b2node = std::static_pointer_cast<slam::eval::Node<BiasType>>(b2_);
                            bias2_->backward(lhs, b2node, jacs);
                            break;
                        }
                        case 8: {
                            auto T_mi_1node = std::static_pointer_cast<slam::eval::Node<PoseType>>(T_mi_1_);
                            transform_i_to_m_1_->backward(lhs, T_mi_1node, jacs);
                            break;
                        }
                        case 9: {
                            auto T_mi_2node = std::static_pointer_cast<slam::eval::Node<PoseType>>(T_mi_2_);
                            transform_i_to_m_2_->backward(lhs, T_mi_2node, jacs);
                            break;
                        }
                    }

                    const auto jacmap = jacs.get();
                    assert(jacmap.size() == 1);
                    keys[i] = jacmap.begin()->first;
                });

                // Hessian and Gradient Update
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