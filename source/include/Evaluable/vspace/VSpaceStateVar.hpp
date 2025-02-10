#pragma once

#include <memory>
#include <Eigen/Core>

#include "source/include/Evaluable/StateVariable.hpp"

namespace slam {
    namespace eval {
        namespace vspace {

            // -----------------------------------------------------------------------------
            /**
             * @class VSpaceStateVar
             * @brief Represents a **state variable in vector space**.
             *
             * **Functionality:**
             * - Stores a vector-valued state variable.
             * - Supports perturbation updates.
             * - Supports cloning.
             *
             * @tparam DIM Dimensionality of the vector space (can be dynamic).
             */
            template <int DIM = Eigen::Dynamic>
            class VSpaceStateVar : public StateVariable<Eigen::Matrix<double, DIM, 1>> {
                public:
                    using Ptr = std::shared_ptr<VSpaceStateVar>;
                    using ConstPtr = std::shared_ptr<const VSpaceStateVar>;

                    using T = Eigen::Matrix<double, DIM, 1>;
                    using Base = StateVariable<T>;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Factory method for creating a shared instance.
                     * @param value Initial state vector.
                     * @param name Optional state variable name.
                     */
                    static Ptr MakeShared(const T& value, const std::string& name = "");

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructor for VSpaceStateVar.
                     * @param value Initial state vector.
                     * @param name Optional state variable name.
                     */
                    explicit VSpaceStateVar(const T& value, const std::string& name = "");

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Updates the state variable using a perturbation.
                     * 
                     * **Update Rule:**  
                     * `x_new = x_old + perturbation`
                     *
                     * @param perturbation The perturbation to apply.
                     * @return `true` if update was successful.
                     * @throws std::runtime_error if the perturbation size is incorrect.
                     */
                    bool update(const Eigen::VectorXd& perturbation) override;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Creates a deep copy of this state variable.
                     * @return A shared pointer to the cloned state variable.
                     */
                    StateVariableBase::Ptr clone() const override;
            };

        }  // namespace vspace
    }  // namespace eval
}  // namespace slam
