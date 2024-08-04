#ifndef REAL_H
#define REAL_H

#include <Kokkos_Core.hpp>

namespace Ibis {

// Alias double as 'real', so we are free to change
// it at a later date if we want FP32 for example
typedef double real;

// Provide a default implementation of mathematic functions
// within the Ibis namespace. Overloads of these functions
// for dual numbers exist in dual.h
template <typename T>
KOKKOS_INLINE_FUNCTION T sqrt(const T& x) {
    return Kokkos::sqrt(x);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T pow(const T& base, const T& power) {
    return Kokkos::pow(base, power);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T max(const T& arg1, const T& arg2) {
    return Kokkos::max(arg1, arg2);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T min(const T& arg1, const T& arg2) {
    return Kokkos::min(arg1, arg2);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T abs(const T& x) {
    return Kokkos::abs(x);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T floor(const T& x) {
    return Kokkos::floor(x);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T ceil(const T& x) {
    return Kokkos::ceil(x);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T copysign(const T& mag, const T& sign) {
    return Kokkos::copysign(mag, sign);
}

KOKKOS_INLINE_FUNCTION real real_part(const real& x) { return x; }

KOKKOS_INLINE_FUNCTION real& real_part(real& x) { return x; }

KOKKOS_INLINE_FUNCTION real dual_part(const real& x) {
    (void)x;
    return 0.0;
}

KOKKOS_INLINE_FUNCTION real dual_part(real& x) {
    (void)x;
    return 0.0;
}

template <typename T>
KOKKOS_INLINE_FUNCTION bool isinf(const T& x) {
    return Kokkos::isinf(x);
}

template <typename T>
KOKKOS_INLINE_FUNCTION bool isnan(const T& x) {
    return Kokkos::isnan(x);
}

}  // namespace Ibis

#endif
