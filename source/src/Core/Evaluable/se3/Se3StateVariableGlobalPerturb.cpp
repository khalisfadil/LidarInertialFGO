#include "Core/Evaluable/se3/Se3StateVariableGlobalPerturb.hpp"

namespace slam {
    namespace eval {
        namespace se3 {

        // -----------------------------------------------------------------------------
        // MakeShared
        // -----------------------------------------------------------------------------

        auto SE3StateVariableGlobalPerturb::MakeShared(const T& value, const std::string& name) -> Ptr {
            return std::make_shared<SE3StateVariableGlobalPerturb>(value, name);
        }

        // -----------------------------------------------------------------------------
        // SE3StateVariableGlobalPerturb
        // -----------------------------------------------------------------------------

        SE3StateVariableGlobalPerturb::SE3StateVariableGlobalPerturb(const T& value, const std::string& name)
            : Base(value, 6, name) {}

        // -----------------------------------------------------------------------------
        // update
        // -----------------------------------------------------------------------------

        bool SE3StateVariableGlobalPerturb::update(const Eigen::VectorXd& perturbation) {
            if (perturbation.size() != this->perturb_dim()) {
                throw std::runtime_error("[SE3StateVariableGlobalPerturb::update] Perturbation size mismatch.");
            }

            // Extract translation and rotation perturbations
            Eigen::Vector3d delta_r = perturbation.head<3>();  // First 3 elements: translation perturbation
            Eigen::Vector3d delta_phi = perturbation.tail<3>(); // Last 3 elements: rotation perturbation

            // Extract current SE(3) transformation components
            Eigen::Matrix4d T_iv = value_.matrix();
            Eigen::Matrix3d C_iv = T_iv.block<3, 3>(0, 0); // Extract rotation matrix
            Eigen::Vector3d r_vi_in_i = T_iv.block<3, 1>(0, 3); // Extract translation vector

            // Apply global perturbation update
            r_vi_in_i += C_iv * delta_r;  // Update translation in global frame
            C_iv *= slam::liemath::so3::vec2rot(delta_phi); // Update rotation using SO(3) exponential map

            // Reconstruct and update transformation
            T_iv.block<3, 3>(0, 0) = C_iv;
            T_iv.block<3, 1>(0, 3) = r_vi_in_i;

            // Explicitly use Eigen::Ref to resolve constructor ambiguity
            value_ = T(Eigen::Ref<const Eigen::Matrix4d>(T_iv));

            return true;
        }

        // -----------------------------------------------------------------------------
        // clone
        // -----------------------------------------------------------------------------

        StateVariableBase::Ptr SE3StateVariableGlobalPerturb::clone() const {
            return std::make_shared<SE3StateVariableGlobalPerturb>(*this);
        }

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
