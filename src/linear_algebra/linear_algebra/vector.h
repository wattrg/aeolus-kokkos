#ifndef LA_VECTOR_H
#define LA_VECTOR_H

#include <util/numeric_types.h>
#include <util/types.h>
#include <Kokkos_Core.hpp>

namespace Ibis {

template <typename T, class Layout = DefaultArrayLayout, class Space = DefaultMemSpace>
using Matrix = Array2D<T, Layout, Space>;

template <typename T, class Layout = DefaultArrayLayout, class Space = DefaultMemSpace>
using Vector = Array1D<T, Layout, Space>;

template <typename T>
auto& column(Matrix<T>& matrix, const size_t col_idx) { 
    return Kokkos::subview(matrix, Kokkos::make_pair(Kokkos::ALL, col_idx)); 
}

template <typename T>
auto& row(Matrix<T>& matrix, const size_t row_idx);

template <typename T>
T norm2(const Vector<T>& vec);

template <typename T>
T norm2_squared(const Vector<T>& vec);

template <typename T>
void scale_in_place(Vector<T>& vec, const T factor);

template <typename T>
void scale(const Vector<T>& vec, Vector<T>& result, const T factor);

template <typename T>
void add_scaled_vector(Vector<T>& vec1, const Vector<T>& vec2, T scale);

template <typename T>
T dot(const Vector<T>& vec1, const Vector<T>& vec2);

} // namespace Ibis

#endif
