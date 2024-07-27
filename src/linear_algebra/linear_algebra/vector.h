#ifndef LA_VECTOR_H
#define LA_VECTOR_H

#include <util/numeric_types.h>
#include <util/types.h>

#include <Kokkos_Core.hpp>

namespace Ibis {

template <typename T, class Layout = DefaultArrayLayout, class Space = DefaultMemSpace>
using Vector = Array1D<T, Layout, Space>;

template <typename T, class Space = DefaultMemSpace, class Layout = DefaultArrayLayout>
class Matrix {
public:
    Matrix() {}
    
    Matrix(std::string name, const size_t n, const size_t m) {
        data_ = Array2D<T, Layout, Space>(name, n, m);
    }

    Matrix(Array2D<T, Layout, Space>& data) : data_(data) {}

    KOKKOS_INLINE_FUNCTION
    T& operator()(const size_t row, const size_t col) { return data_(row, col); }

    KOKKOS_INLINE_FUNCTION
    T& operator()(const size_t row, const size_t col) const { return data_(row, col); }

    void set_to_identity() {
        assert(data_.extent(0) == data_.extent(1));
        auto data = data_;
        Kokkos::deep_copy(data_, T(0.0));
        Kokkos::parallel_for(
            "Matrix::set_to_identity", data_.extent(0),
            KOKKOS_LAMBDA(const size_t i) { data(i, i) = T(1.0); });
    }

    // Return a sub-matrix. The data is the same data, so any modifications
    // to either matrix will be seen in the other matrix
    Matrix<T, Space, Kokkos::LayoutStride> sub_matrix(const size_t start_row,
                                                      const size_t end_row,
                                                      const size_t start_col,
                                                      const size_t end_col) {
        return Matrix<T, Space, Kokkos::LayoutStride>(
            data_, Kokkos::make_pair(start_row, end_row),
            Kokkos::make_pair(start_col, end_col));
    }

    // The row vector at a given row in the matrix
    // The data is the same, so any changes to the vector or matrix
    // will appear in the other
    Vector<T, Kokkos::LayoutStride, Space> row(const size_t row) {
        return Vector<T, Kokkos::LayoutStride, Space>(
            Kokkos::subview(data_, row, Kokkos::ALL));
    }

    // The column vector at a given column in the matrix
    // The data is the same, so any changes to the vector or matrix
    // will appear in the other
    Vector<T, Kokkos::LayoutStride, Space> column(const size_t col) {
        return Vector<T, Kokkos::LayoutStride, Space>(
            Kokkos::subview(data_, Kokkos::ALL, col));
    }

private:
    Array2D<T, Layout, Space> data_;
};

// template <typename T, class Layout = DefaultArrayLayout, class Space = DefaultMemSpace>
// using Matrix = Array2D<T, Layout, Space>;

template <typename T>
auto& row(Matrix<T>& matrix, const size_t row_idx);

template <typename T>
T norm2(const Vector<T>& vec);

template <typename T>
T norm2_squared(const Vector<T>& vec);

template <typename T>
void scale_in_place(Vector<T>& vec, const T factor);

template <typename T, class Space, class Layout1, class Layout2>
void scale(const Vector<T>& vec, Vector<T>& result, const T factor) {
    assert(vec.extent(0) == result.extent(0));
    Kokkos::parallel_for(
        "Ibis::Vector::scale", vec.extent(0),
        KOKKOS_LAMBDA(const size_t i) { result(i) = vec(i) * factor; });
}

template <typename T, class Space, class Layout1, class Layout2>
void add_scaled_vector(Vector<T, Layout1, Space>& vec1,
                       const Vector<T, Layout2, Space>& vec2, T scale) {
    assert(vec1.extent(0) == vec2.extent(0));
    Kokkos::parallel_for(
        "Ibis::Vector::subtract_scated_vector", vec1.extent(0),
        KOKKOS_LAMBDA(const size_t i) { vec1(i) += vec2(i) * scale; });
}

template <typename T, class Space, class Layout1, class Layout2>
void deep_copy_vector(Vector<T, Layout1, Space>& dest, const Vector<T, Layout2, Space>& src) {
    assert(dest.extent(0) == src.extent(0));
    // For the moment we'll make this general. If Layout1 and Layout2 are different
    // we cannot use Kokkos::deep_copy. This is the intended use of this function.
    // We could detect if Layout1 and Layout2 are the same, but meh...
    Kokkos::parallel_for("Ibis::deep_copy_vector", dest.extent(0),
                         KOKKOS_LAMBDA(const size_t i) {
            dest(i) = src(i);            
        }
    );
}

template <typename T>
T dot(const Vector<T>& vec1, const Vector<T>& vec2);

}  // namespace Ibis

#endif
