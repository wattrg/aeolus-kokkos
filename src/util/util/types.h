#ifndef TYPES_H
#define TYPES_H

#include <Kokkos_Core.hpp>

namespace Ibis {

using DefaultExecSpace = Kokkos::DefaultExecutionSpace;
using DefaultHostExecSpace = Kokkos::DefaultHostExecutionSpace;
using DefaultArrayLayout = Kokkos::DefaultExecutionSpace::array_layout;
using DefaultHostArrayLayout = Kokkos::DefaultHostExecutionSpace::array_layout;
using DefaultMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using DefaultHostMemSpace = Kokkos::DefaultHostExecutionSpace::memory_space;

// A dynamically sized one dimensional array
template <typename T, class Layout = DefaultArrayLayout, class Space = DefaultMemSpace>
using Array1D = Kokkos::View<T*, Layout, Space>;

// A dynaically sized two dimensional array
template <typename T, class Layout = DefaultArrayLayout, class Space = DefaultMemSpace>
using Array2D = Kokkos::View<T**, Layout, Space>;

// A view into a 2D array
template <typename T, class Space = DefaultMemSpace>
using SubArray2D = Kokkos::View<T*, Kokkos::LayoutStride, Space>;

// An n x 3 array
template <typename T, class Layout = Kokkos::DefaultExecutionSpace::array_layout,
          class Space = Kokkos::DefaultExecutionSpace::memory_space>
using ArrayNby3 = Kokkos::View<T* [3], Layout, Space>;

template <typename T, class Layout = DefaultArrayLayout, class Space = DefaultMemSpace>
using Vector = Array1D<T, Layout, Space>;

template <typename T, class Layout = DefaultArrayLayout, class Space = DefaultMemSpace>
using Matrix = Array2D<T, Layout, Space>;

}  // namespace Ibis

#endif
