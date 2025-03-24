#pragma once

namespace slam {

    /**
     * @brief 3D point structure with spatial coordinates, attributes, and timing information.
     */
    enum class ColorMode {
                Occupancy,    // Based on point counter
                Reflectivity, // Based on average reflectivity
                Intensity,    // Based on average intensity
                NIR           // Based on near-infrared value
            };

}  // namespace slam