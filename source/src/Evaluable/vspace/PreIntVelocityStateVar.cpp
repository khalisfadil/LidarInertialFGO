#include "source/include/Evaluable/vspace/PreIntVelocityStateVar.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // ----------------------------------------------------------------------------
            // MakeShared
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto PreIntVelocityStateVar<DIM>::MakeShared(
                const T& value, 
                const Evaluable<InType>::ConstPtr& T_iv, 
                const std::string& name) -> Ptr {
                return std::make_shared<PreIntVelocityStateVar<DIM>>(value, T_iv, name);
            }

            // ----------------------------------------------------------------------------
            // PreIntVelocityStateVar Constructor
            // ----------------------------------------------------------------------------

            template <int DIM>
            PreIntVelocityStateVar<DIM>::PreIntVelocityStateVar(
                const T& value, 
                const Evaluable<InType>::ConstPtr& T_iv, 
                const std::string& name)
                : Base(value, DIM, name), T_iv_(T_iv) {}

            // ----------------------------------------------------------------------------
            // update
            // ----------------------------------------------------------------------------

            template <int DIM>
            bool PreIntVelocityStateVar<DIM>::update(const Eigen::VectorXd& perturbation) {
                if (perturbation.size() != this->perturb_dim()) {
                    throw std::runtime_error(
                        "[PreIntVelocityStateVar::update] Perturbation size mismatch.");
                }

                // Extract rotation matrix from transformation T_iv efficiently
                const Eigen::Matrix3d C_iv = T_iv_->value().rotation();

                // Store original value in case of exception
                T original_value = this->value_;

                try {
                    // Update velocity state
                    this->value_ += C_iv * perturbation;
                    return true;
                } catch (...) {
                    // Revert to original state if an error occurs
                    this->value_ = original_value;
                    throw;
                }
            }

            // ----------------------------------------------------------------------------
            // clone
            // ----------------------------------------------------------------------------

            template <int DIM>
            StateVariableBase::Ptr PreIntVelocityStateVar<DIM>::clone() const {
                return std::make_shared<PreIntVelocityStateVar<DIM>>(*this);
            }

            // Explicit template instantiations
            template class PreIntVelocityStateVar<1>;
            template class PreIntVelocityStateVar<2>;
            template class PreIntVelocityStateVar<3>;

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam
