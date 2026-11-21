// The limit the drag kernel clips at; a stiff model clips earlier.
#pragma once

namespace physics::detail {

#if defined(MODEL_STIFF)
inline constexpr double kLimit = 4.0;
#else
inline constexpr double kLimit = 2.0;
#endif

} // namespace physics::detail
