#pragma once

#include <cstdint>
#include <functional>
#include <tbb/concurrent_hash_map.h>

namespace slam {
    namespace traj {

            // -----------------------------------------------------------------------------
            /**
             * @class Time
             * @brief Represents a high-precision timestamp using nanoseconds.
             *
             * The `Time` class provides a representation of time in **nanosecond precision**,
             * supporting arithmetic operations and conversions to seconds.
             * 
             * This is particularly useful in **trajectory estimation, SLAM, and sensor fusion**,
             * where high-precision timestamps are required for synchronization.
             */
            class Time {
                public:
                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Default constructor, initializes time to zero.
                     */
                    Time() : nsecs_(0) {}

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructs a time object from nanoseconds.
                     * @param nsecs Time in nanoseconds.
                     */
                    explicit Time(int64_t nsecs) : nsecs_(nsecs) {}

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructs a time object from seconds.
                     * @param secs Time in seconds (converted to nanoseconds).
                     */
                    explicit Time(double secs) : nsecs_(secs * 1e9) {}

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Constructs a time object from separate seconds and nanoseconds.
                     * @param secs  Number of seconds.
                     * @param nsec  Additional nanoseconds.
                     */
                    Time(int32_t secs, int32_t nsec) {
                        nsecs_ = static_cast<int64_t>(secs) * 1'000'000'000 + static_cast<int64_t>(nsec);
                    }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Converts time to seconds (double precision).
                     * @return Time in seconds.
                     */
                    double seconds() const { return static_cast<double>(nsecs_) * 1e-9; }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Retrieves the time in nanoseconds.
                     * @return Time in nanoseconds.
                     */
                    const int64_t& nanosecs() const { return nsecs_; }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Adds another `Time` object to this one.
                     * @param other The other `Time` object.
                     * @return Reference to updated `Time` object.
                     */
                    Time& operator+=(const Time& other);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Adds two `Time` objects.
                     * @param other The other `Time` object.
                     * @return The sum of two time instances.
                     */
                    Time operator+(const Time& other) const;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Subtracts another `Time` object from this one.
                     * @param other The other `Time` object.
                     * @return Reference to updated `Time` object.
                     */
                    Time& operator-=(const Time& other);

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Subtracts two `Time` objects.
                     * @param other The other `Time` object.
                     * @return The difference between two time instances.
                     */
                    Time operator-(const Time& other) const;

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Equality comparison operator.
                     */
                    bool operator==(const Time& other) const { return nsecs_ == other.nsecs_; }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Inequality comparison operator.
                     */
                    bool operator!=(const Time& other) const { return nsecs_ != other.nsecs_; }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Less than comparison operator.
                     */
                    bool operator<(const Time& other) const { return nsecs_ < other.nsecs_; }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Greater than comparison operator.
                     */
                    bool operator>(const Time& other) const { return nsecs_ > other.nsecs_; }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Less than or equal comparison operator.
                     */
                    bool operator<=(const Time& other) const { return nsecs_ <= other.nsecs_; }

                    // -----------------------------------------------------------------------------
                    /**
                     * @brief Greater than or equal comparison operator.
                     */
                    bool operator>=(const Time& other) const { return nsecs_ >= other.nsecs_; }

                private:
                    /**
                     * @brief Stores time in nanoseconds for high precision.
                     * 
                     * Using `int64_t` allows nanosecond precision, ensuring:
                     * - Sufficient range for **sensor timestamps**.
                     * - Compatibility with **trajectory estimation and mapping**.
                     */
                    int64_t nsecs_;
                };

            // -----------------------------------------------------------------------------
            /**
             * @brief Concurrent hash map for `Time`, using Intel TBB for thread-safe operations.
             */
            struct TimeHashCompare {
                static size_t hash(const Time& t) {
                    return std::hash<int64_t>{}(t.nanosecs());
                }

                static bool equal(const Time& t1, const Time& t2) {
                    return t1 == t2;
                }
            };

            using ConcurrentTimeHashMap = tbb::concurrent_hash_map<Time, std::string, TimeHashCompare>;

    }  // namespace traj
}  // namespace slam
