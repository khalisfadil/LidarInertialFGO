#pragma once

#include <include/LieGroupMath/LieGroupMath.hpp>
#include <include/Evaluable/StateVariable.hpp>

namespace slam {
    namespace eval {
        namespace se3 {

            // -----------------------------------------------------------------------------
            /**
             * @class SE3StateVariable
             * @brief Represents an SE(3) transformation as a state variable for optimization.
             *
             * This class encapsulates an SE(3) transformation and allows perturbations
             * to be applied in Lie algebra space using left-multiplicative updates:
             * \f[
             * T' = \exp(\hat{\xi}) T
             * \f]
             * where \( \hat{\xi} \) is a 6D twist vector in \( \mathfrak{se}(3) \).
             * This approach ensures smooth updates in non-linear optimization problems.
             */
            class SE3StateVariable : public StateVariable<slam::liemath::se3::Transformation> {
                public:
                    using Ptr = std::shared_ptr<SE3StateVariable>;
                    using ConstPtr = std::shared_ptr<const SE3StateVariable>;

                    using T = slam::liemath::se3::Transformation;
                    using Base = StateVariable<T>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method to create a shared pointer instance.
                     * @param value Initial SE(3) transformation.
                     * @param name Optional state variable name.
                     * @return Shared pointer to a new SE3StateVariable instance.
                     */
                    static Ptr MakeShared(const T& value, const std::string& name = "");

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructs an SE3StateVariable with an initial transformation.
                     * @param value Initial SE(3) transformation.
                     * @param name Optional state variable name.
                     */
                    SE3StateVariable(const T& value, const std::string& name = "");

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Updates the transformation using a Lie algebra perturbation.
                     * 
                     * This applies an incremental transformation using the exponential map:
                     * \f[
                     * T' = \exp(\hat{\xi}) T
                     * \f]
                     * 
                     * @param perturbation 6D perturbation vector \( \xi \).
                     * @return True if the update is successful.
                     * @throws std::runtime_error if the perturbation vector has an incorrect size.
                     */
                    bool update(const Eigen::VectorXd& perturbation) override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Creates an independent copy of this state variable.
                     * @return Shared pointer to a cloned SE3StateVar.
                     */
                    StateVariableBase::Ptr clone() const override;
            };
        }  // namespace se3
    }  // namespace eval
}  // namespace slam
