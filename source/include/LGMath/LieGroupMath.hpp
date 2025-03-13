#pragma once

// ==================== SO(2): Special Orthogonal Group in 2D ====================
// TODO: Implement and include SO(2) components when available

// ==================== SE(2): Special Euclidean Group in 2D ====================
// TODO: Implement and include SE(2) components when available

// ==================== SO(3): Special Orthogonal Group in 3D ====================
#include "LGMath/so3/Operations.hpp"      ///< Operations for SO(3) rotations
#include "LGMath/so3/Rotations.hpp"       ///< SO(3) rotation representation
#include "LGMath/so3/Types.hpp"           ///< Type definitions for SO(3)

// ==================== SE(3): Special Euclidean Group in 3D ====================
#include "LGMath/se3/Operations.hpp"       ///< Operations for SE(3) transformations
#include "LGMath/se3/Transformation.hpp"   ///< SE(3) transformation representation
#include "LGMath/se3/Types.hpp"            ///< Type definitions for SE(3)
#include "LGMath/se3/TransformationWithCovariance.hpp" 

// ==================== R^3: Euclidean Space in 3D ====================
#include "LGMath/r3/Operations.hpp"       ///< Operations for R^3 vector space
#include "LGMath/r3/Types.hpp"            ///< Type definitions for R^3


