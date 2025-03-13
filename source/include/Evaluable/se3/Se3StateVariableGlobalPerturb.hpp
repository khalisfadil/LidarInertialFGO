#pragma once

#include "LGMath/LieGroupMath.hpp"
#include "Evaluable/StateVariable.hpp"

namespace slam {
    namespace eval {
        namespace se3 {

            // -----------------------------------------------------------------------------
            /**
             * @class SE3StateVariableGlobalPerturb
             * @brief Represents an SE(3) transformation as a state variable with global perturbations.
             *
             * This class encapsulates an SE(3) transformation and allows perturbations
             * to be applied **directly in the global frame** using:
             * \f[
             * C \leftarrow C \cdot \exp(\hat{\delta \phi})
             * \f]
             * \f[
             * r \leftarrow r + C \cdot \delta r
             * \f]
             * where \( \delta \phi \) and \( \delta r \) are small rotational and translational perturbations.
             * This update method is particularly useful in **global optimization** and **loop closure constraints**.
             */
            class SE3StateVariableGlobalPerturb : public StateVariable<slam::liemath::se3::Transformation> {
            public:
                using Ptr = std::shared_ptr<SE3StateVariableGlobalPerturb>;
                using ConstPtr = std::shared_ptr<const SE3StateVariableGlobalPerturb>;

                using T = slam::liemath::se3::Transformation;
                using Base = StateVariable<T>;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Factory method to create a shared pointer instance.
                 * @param value Initial SE(3) transformation.
                 * @param name Optional state variable name.
                 * @return Shared pointer to a new SE3StateVariableGlobalPerturb instance.
                 */
                static Ptr MakeShared(const T& value, const std::string& name = "");

                // -----------------------------------------------------------------------------
                /**
                 * @brief Constructs an SE3StateVariableGlobalPerturb with an initial transformation.
                 * @param value Initial SE(3) transformation.
                 * @param name Optional state variable name.
                 */
                SE3StateVariableGlobalPerturb(const T& value, const std::string& name = "");

                // -----------------------------------------------------------------------------
                /**
                 * @brief Updates the transformation using a global perturbation.
                 *
                 * This applies an incremental transformation in the **global frame**:
                 * \f[
                 * C \leftarrow C \cdot \exp(\hat{\delta \phi})
                 * \f]
                 * \f[
                 * r \leftarrow r + C \cdot \delta r
                 * \f]
                 * 
                 * @param perturbation 6D perturbation vector \( \xi = [\delta r, \delta \phi] \).
                 * @return True if the update is successful.
                 * @throws std::runtime_error if the perturbation vector has an incorrect size.
                 */
                bool update(const Eigen::VectorXd& perturbation) override;

                // -----------------------------------------------------------------------------
                /**
                 * @brief Creates an independent copy of this state variable.
                 * @return Shared pointer to a cloned SE3StateVarGlobalPerturb.
                 */
                StateVariableBase::Ptr clone() const override;
            };

        }  // namespace se3
    }  // namespace eval
}  // namespace slam
