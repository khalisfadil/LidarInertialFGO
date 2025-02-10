#pragma once

#include <chrono>

namespace slam {
    namespace common {
        
        // -----------------------------------------------------------------------------
        /**
         * @class Timer
         * @brief A simple high-resolution timer for measuring elapsed time.
         */
        class Timer {
            public:
                // -----------------------------------------------------------------------------
                /** @brief Default constructor initializes and starts the timer. */
                Timer() { reset(); }

                // -----------------------------------------------------------------------------
                /** @brief Resets the timer to start counting from zero. */
                void reset() {
                    start_time_ = std::chrono::high_resolution_clock::now();
                }

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Returns the elapsed time in seconds since the last reset.
                 * @return Elapsed time in seconds.
                 */
                [[nodiscard]] double seconds() const {
                    return std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - start_time_
                    ).count();
                }

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Returns the elapsed time in milliseconds since the last reset.
                 * @return Elapsed time in milliseconds.
                 */
                [[nodiscard]] double milliseconds() const {
                    return seconds() * 1000.0;
                }

            private:
                // -----------------------------------------------------------------------------
                /** @brief Stores the starting time point */
                std::chrono::high_resolution_clock::time_point start_time_;
        };

    } // namespace common
} // namespace slam
