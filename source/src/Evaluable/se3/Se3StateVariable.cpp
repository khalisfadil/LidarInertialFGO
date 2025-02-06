#include "source/include/Evaluable/se3/Se3StateVariable.hpp"

namespace slam {
    namespace eval {
        namespace se3 {

            // -----------------------------------------------------------------------------
            // MakeShared
            // -----------------------------------------------------------------------------

            auto SE3StateVariable::MakeShared(const T& value, const std::string& name) -> Ptr {
                return std::make_shared<SE3StateVariable>(value, name);
            }

            // -----------------------------------------------------------------------------
            // SE3StateVariable
            // -----------------------------------------------------------------------------

            SE3StateVariable::SE3StateVariable(const T& value, const std::string& name)
                : Base(value, 6, name) {}

            // -----------------------------------------------------------------------------
            // update
            // -----------------------------------------------------------------------------

            bool SE3StateVariable::update(const Eigen::VectorXd& perturbation) {
                if (perturbation.size() != this->perturb_dim()) {
                    throw std::runtime_error("[SE3StateVariable::update] Perturbation size mismatch.");
                }

                // Ensure perturbation is interpreted as a 6D twist vector
                value_ = T(Eigen::Ref<const Eigen::VectorXd>(perturbation)) * value_;
                return true;
            }

            // -----------------------------------------------------------------------------
            // clone
            // -----------------------------------------------------------------------------

            StateVariableBase::Ptr SE3StateVariable::clone() const {
                return std::make_shared<SE3StateVariable>(*this);
            }

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
