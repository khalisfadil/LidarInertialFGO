#pragma once

#include <ctime>
#include <ratio>
#include <chrono>

namespace slam {
    namespace common {
        
        // -----------------------------------------------------------------------------
        /**
         * @class Timer
         * @brief A high-resolution POSIX timer for measuring elapsed time.
         * 
         * Uses `clock_gettime(CLOCK_MONOTONIC, &timespec)` for **nanosecond precision**.
         * - **Thread-safe**
         * - **Unaffected by system clock changes**
         * - **Better resolution than `gettimeofday()`**
         */
        class Timer {
            public:
                // -----------------------------------------------------------------------------
                /** @brief Default constructor initializes and starts the timer. */
                Timer() { reset(); }

                // -----------------------------------------------------------------------------
                /** @brief Resets the timer to start counting from zero. */
                void reset() {
                    start_time_ = getCurrentTime();
                }

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Returns the elapsed time in seconds since the last reset.
                 * @return Elapsed time in seconds.
                 */
                [[nodiscard]] double seconds() const {
                    return getElapsedTime().count();
                }

                // -----------------------------------------------------------------------------
                /** 
                 * @brief Returns the elapsed time in milliseconds since the last reset.
                 * @return Elapsed time in milliseconds.
                 */
                [[nodiscard]] double milliseconds() const {
                    return getElapsedTime().count() * 1000.0;
                }

            private:
                // -----------------------------------------------------------------------------
                /** @brief Stores the starting time point */
                struct timespec start_time_;

                // -----------------------------------------------------------------------------
                /** @brief Retrieves the current time using `clock_gettime()` */
                static struct timespec getCurrentTime() {
                    struct timespec ts{};
                    clock_gettime(CLOCK_MONOTONIC, &ts);  // High-resolution, monotonic time
                    return ts;
                }

                // -----------------------------------------------------------------------------
                /** @brief Computes elapsed time since reset */
                [[nodiscard]] std::chrono::duration<double> getElapsedTime() const {
                    struct timespec now = getCurrentTime();
                    double elapsed_sec = (now.tv_sec - start_time_.tv_sec) +
                                         (now.tv_nsec - start_time_.tv_nsec) * 1e-9;
                    return std::chrono::duration<double>(elapsed_sec);
                }
        };

    } // namespace common
} // namespace slam
