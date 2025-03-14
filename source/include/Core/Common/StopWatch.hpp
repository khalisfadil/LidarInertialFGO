#pragma once

#include <tbb/tick_count.h>
#include <mutex>
#include <iostream>

namespace slam {
    namespace common {

        // -----------------------------------------------------------------------------
        /**
         * @class Stopwatch
         * @brief A thread-safe stopwatch using Intel TBB for high-performance timing.
         * 
         * Provides precise timing using `tbb::tick_count`, optimized for parallelism.
         */
        class Stopwatch {
            public:
                // -----------------------------------------------------------------------------
                /** @brief Constructor with optional immediate start. */
                explicit Stopwatch(bool start = true) {
                    if (start) this->start();
                }

                // -----------------------------------------------------------------------------
                /** @brief Starts or resumes the stopwatch. */
                void start() {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (!started_) {
                        started_ = true;
                        paused_ = false;
                        reference_ = tbb::tick_count::now();
                        accumulated_ = 0.0;
                    } else if (paused_) {
                        paused_ = false;
                        reference_ = tbb::tick_count::now();
                    }
                }

                // -----------------------------------------------------------------------------
                /** @brief Pauses the stopwatch. */
                void stop() {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (started_ && !paused_) {
                        accumulated_ += (tbb::tick_count::now() - reference_).seconds();
                        paused_ = true;
                    }
                }

                // -----------------------------------------------------------------------------
                /** @brief Resets the stopwatch to zero. */
                void reset() {
                    std::lock_guard<std::mutex> lock(mutex_);
                    started_ = false;
                    paused_ = false;
                    reference_ = tbb::tick_count::now();
                    accumulated_ = 0.0;
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Returns the elapsed time in **seconds**.
                 * @return Elapsed time as a `double` (in seconds).
                 */
                [[nodiscard]] double seconds() const {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (!started_) return 0.0;
                    return accumulated_ + (paused_ ? 0.0 : (tbb::tick_count::now() - reference_).seconds());
                }

                // -----------------------------------------------------------------------------
                /**
                 * @brief Returns the elapsed time in **milliseconds**.
                 * @return Elapsed time as a `double` (in milliseconds).
                 */
                [[nodiscard]] double milliseconds() const {
                    return seconds() * 1000.0;
                }

                // -----------------------------------------------------------------------------
                /** @brief Overloads `<<` for easy output. */
                friend std::ostream& operator<<(std::ostream& os, const Stopwatch& sw) {
                    return os << sw.milliseconds() << "ms";
                }

            private:
                mutable std::mutex mutex_;  ///< Mutex for thread safety
                bool started_ = false;      ///< Whether the stopwatch is started
                bool paused_ = false;       ///< Whether the stopwatch is paused
                tbb::tick_count reference_ = tbb::tick_count::now();  ///< Reference start time
                double accumulated_ = 0.0;  ///< Accumulated time (in seconds)
        };

    }  // namespace common
}  // namespace slam
