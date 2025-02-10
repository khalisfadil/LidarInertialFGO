#include "source/include/Evaluable/vspace/VSpaceStateVar.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // ----------------------------------------------------------------------------
            // MakeShared
            // ----------------------------------------------------------------------------

            template <int DIM>
            auto VSpaceStateVar<DIM>::MakeShared(const T& value, const std::string& name) -> Ptr {
                return std::make_shared<VSpaceStateVar<DIM>>(value, name);
            }

            // ----------------------------------------------------------------------------
            // VSpaceStateVar Constructor
            // ----------------------------------------------------------------------------

            template <int DIM>
            VSpaceStateVar<DIM>::VSpaceStateVar(const T& value, const std::string& name)
                : Base(value, DIM, name) {}

            // ----------------------------------------------------------------------------
            // update
            // ----------------------------------------------------------------------------

            template <int DIM>
            bool VSpaceStateVar<DIM>::update(const Eigen::VectorXd& perturbation) {
                if (perturbation.size() != this->perturb_dim()) {
                    throw std::runtime_error("[VSpaceStateVar::update] Perturbation size mismatch.");
                }

                // Store original value in case of exception
                T original_value = this->value_;

                try {
                    // Apply perturbation
                    this->value_ += perturbation;
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
            StateVariableBase::Ptr VSpaceStateVar<DIM>::clone() const {
                return std::make_shared<VSpaceStateVar<DIM>>(*this);
            }

            // ----------------------------------------------------------------------------
            // Explicit template instantiations
            // ----------------------------------------------------------------------------

            template class VSpaceStateVar<1>;
            template class VSpaceStateVar<2>;
            template class VSpaceStateVar<3>;

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam
