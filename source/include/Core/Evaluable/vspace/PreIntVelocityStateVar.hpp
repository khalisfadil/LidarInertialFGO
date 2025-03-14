#pragma once

#include <memory>
#include <Eigen/Core>

#include "LGMath/LieGroupMath.hpp"  // Updated to liemath
#include "Core/Evaluable/StateVariable.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // -----------------------------------------------------------------------------
            /**
             * @class PreIntVelocityStateVar
             * @brief Represents a **pre-integrated velocity state variable**.
             *
             * **Functionality:**
             * - Represents velocity pre-integrated over time.
             * - Uses a transformation `T_iv` to update velocity.
             * - Supports cloning and perturbation updates.
             *
             * @tparam DIM Dimensionality of the velocity vector (e.g., 3 for 3D).
             */
            template <int DIM = Eigen::Dynamic>
            class PreIntVelocityStateVar : public StateVariable<Eigen::Matrix<double, DIM, 1>> {
            public:
                using Ptr = std::shared_ptr<PreIntVelocityStateVar>;
                using ConstPtr = std::shared_ptr<const PreIntVelocityStateVar>;

                using T = Eigen::Matrix<double, DIM, 1>;
                using Base = StateVariable<T>;
                using InType = slam::liemath::se3::Transformation;  // Namespace updated

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method for creating a shared instance.
                 * @param value Initial velocity estimate.
                 * @param T_iv Transformation used for velocity update.
                 * @param name Optional state variable name.
                 */
                static Ptr MakeShared(const T& value, 
                                      const Evaluable<InType>::ConstPtr& T_iv, 
                                      const std::string& name = "") {
                    return std::make_shared<PreIntVelocityStateVar<DIM>>(value, T_iv, name);
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructor for PreIntVelocityStateVar.
                 * @param value Initial velocity estimate.
                 * @param T_iv Transformation used for velocity update.
                 * @param name Optional state variable name.
                 */
                PreIntVelocityStateVar(const T& value, 
                                       const Evaluable<InType>::ConstPtr& T_iv, 
                                       const std::string& name = "")
                    : Base(value, DIM, name), T_iv_(T_iv) {}

                // -----------------------------------------------------------------------------
                /**
                 * @brief Updates the velocity state using a perturbation.
                 * 
                 * **Update Rule:**  
                 * `v_new = v_old + C_iv * perturbation`
                 * where `C_iv` is the rotation matrix from transformation `T_iv`.
                 *
                 * @param perturbation The perturbation to apply.
                 * @return `true` if update was successful, `false` otherwise.
                 * @throws std::runtime_error if the perturbation size is incorrect.
                 */
                bool update(const Eigen::VectorXd& perturbation) override {
                    // Validate perturbation size
                    if (perturbation.size() != this->perturb_dim()) {
                        throw std::runtime_error("[PreIntVelocityStateVar::update] Perturbation size mismatch.");
                    }

                    // Extract rotation matrix reference
                    const Eigen::Matrix3d& C_iv = T_iv_->value().matrix().block<3, 3>(0, 0);

                    // Efficient in-place update
                    this->value_ += C_iv * perturbation;

                    return true;
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Creates a deep copy of this state variable.
                 * @return A shared pointer to the cloned state variable.
                 */
                StateVariableBase::Ptr clone() const override {
                    return std::make_shared<PreIntVelocityStateVar<DIM>>(*this);
                }

            private:
                const Evaluable<InType>::ConstPtr T_iv_;  ///< Transformation for velocity updates.
            };

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam