#include "source/include/Trajectory/Time.hpp"


namespace slam {
    namespace traj {

        // -----------------------------------------------------------------------------
        // operator+=
        // -----------------------------------------------------------------------------

        Time& Time::operator+=(const Time& other) {
            nsecs_ += other.nsecs_;
            return *this;
        }

        // -----------------------------------------------------------------------------
        // operator+
        // -----------------------------------------------------------------------------

        Time Time::operator+(const Time& other) const {
            Time temp(*this);
            temp += other;
            return temp;
        }

        // -----------------------------------------------------------------------------
        // operator-=
        // -----------------------------------------------------------------------------

        Time& Time::operator-=(const Time& other) {
            nsecs_ -= other.nsecs_;
            return *this;
        }

        // -----------------------------------------------------------------------------
        // operator-
        // -----------------------------------------------------------------------------

        Time Time::operator-(const Time& other) const {
            Time temp(*this);
            temp -= other;
            return temp;
        }

    }  // namespace traj
}  // namespace slam
