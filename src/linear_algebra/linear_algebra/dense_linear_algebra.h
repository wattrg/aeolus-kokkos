#ifndef DENSE_LINEAR_ALGEBRA_H
#define DENSE_LINEAR_ALGEBRA_H

#include <util/numeric_types.h>
#include <util/types.h>

#include <Kokkos_Core.hpp>

namespace Ibis {

// template <typename T, class Layout = DefaultArrayLayout, class Space = DefaultMemSpace>
// using Vector = Array1D<T, Layout, Space>;

template <typename T, class ExecSpace = DefaultExecSpace,
          class Layout = typename ExecSpace::array_layout,
          class MemSpace = typename ExecSpace::memory_space>
class Vector {
public:
    Vector() {}

    Vector(std::string name, size_t n_values) {
        data_ = Array1D<T, Layout, MemSpace>(name, n_values);
    }

    Vector(Array1D<T, Layout, MemSpace> data) : data_(data) {}

    KOKKOS_INLINE_FUNCTION
    T& operator()(const size_t i) { return data_(i); }

    KOKKOS_INLINE_FUNCTION
    T& operator()(const size_t i) const { return data_(i); }

    Vector<T, ExecSpace, Kokkos::LayoutStride, MemSpace> sub_vector(const size_t start,
                                                                    const size_t end) {
        assert(end <= size() + 1);
        
        return Vector<T, ExecSpace, Kokkos::LayoutStride, MemSpace>(
            Kokkos::subview(data_, Kokkos::make_pair(start, end)));
    }

    // this deep_copy overload copies a vector from one memory space to another
    // as long as they have the same array layout
    template <class OtherExecSpace, class OtherMemSpace>
    void deep_copy_space(Vector<T, OtherExecSpace, Layout, OtherMemSpace>& other) {
        Kokkos::deep_copy(data_, other.data());
    }

    // this deep_copy overload copies a vector from one layout to another layout
    // as long as they are in the same memory space
    template <class OtherLayout>
    void deep_copy_layout(Vector<T, ExecSpace, OtherLayout, MemSpace>& other) {
        assert(size() == other.size());
        // For the moment we'll make this general. If Layout1 and Layout2 are different
        // we cannot use Kokkos::deep_copy. This is the intended use of this function.
        // We could detect if Layout1 and Layout2 are the same, but meh...
        auto data = data_;
        Kokkos::parallel_for(
            "Ibis::deep_copy_vector", Kokkos::RangePolicy<ExecSpace>(0, other.size()),
            KOKKOS_LAMBDA(const size_t i) { data(i) = other(i); });
    }

    KOKKOS_INLINE_FUNCTION
    size_t size() const { return data_.extent(0); }

    Array1D<T, Layout, MemSpace> data() { return data_; }

private:
    Array1D<T, Layout, MemSpace> data_;
};

template <typename T, class ExecSpace = DefaultExecSpace,
          class Layout = typename ExecSpace::array_layout,
          class MemSpace = typename ExecSpace::memory_space>
class Matrix {
public:
    Matrix() {}

    Matrix(std::string name, const size_t n, const size_t m) {
        data_ = Array2D<T, Layout, MemSpace>(name, n, m);
    }

    Matrix(Array2D<T, Layout, MemSpace> data) : data_(data) {}

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
    Matrix<T, ExecSpace, Kokkos::LayoutStride, MemSpace> sub_matrix(
        const size_t start_row, const size_t end_row, const size_t start_col,
        const size_t end_col) {
        assert(end_row <= n_rows() + 1);
        assert(end_col <= n_cols() + 1);
        
        return Matrix<T, ExecSpace, Kokkos::LayoutStride, MemSpace>(
            Kokkos::subview(data_, Kokkos::make_pair(start_row, end_row),
                            Kokkos::make_pair(start_col, end_col)));
    }

    // Return a sub-matrix containing the columns i->j
    Matrix<T, ExecSpace, Kokkos::LayoutStride, MemSpace> columns(const size_t start_col,
                                                                 const size_t end_col) {
        return Matrix<T, ExecSpace, Kokkos::LayoutStride, MemSpace>(
            Kokkos::subview(data_, Kokkos::ALL, Kokkos::make_pair(start_col, end_col)));
    }

    // The row vector at a given row in the matrix
    // The data is the same, so any changes to the vector or matrix
    // will appear in the other
    Vector<T, ExecSpace, Kokkos::LayoutStride, MemSpace> row(const size_t row) {
        return Vector<T, ExecSpace, Kokkos::LayoutStride, MemSpace>(
            Kokkos::subview(data_, row, Kokkos::ALL));
    }

    // The column vector at a given column in the matrix
    // The data is the same, so any changes to the vector or matrix
    // will appear in the other
    Vector<T, ExecSpace, Kokkos::LayoutStride, MemSpace> column(const size_t col) {
        return Vector<T, ExecSpace, Kokkos::LayoutStride, MemSpace>(
            Kokkos::subview(data_, Kokkos::ALL, col));
    }

    template <typename OtherLayout>
    void deep_copy(Matrix<T, ExecSpace, OtherLayout, MemSpace>& other) {
        assert(this->n_rows() == other.n_rows());
        assert(this->n_cols() == other.n_rows());

        size_t n_rows = this->n_rows();
        size_t n_cols = this->n_cols();

        auto data = data_;
        Kokkos::parallel_for(
            "Matrix::deep_copy", n_rows, KOKKOS_LAMBDA(const size_t row) {
                for (size_t col = 0; col < n_cols; col++) {
                    data(row, col) = other(row, col);
                }
            });
    }

    KOKKOS_INLINE_FUNCTION
    size_t n_rows() const { return data_.extent(0); }

    KOKKOS_INLINE_FUNCTION
    size_t n_cols() const { return data_.extent(1); }

private:
    Array2D<T, Layout, MemSpace> data_;
};

template <typename T, class ExecSpace, class Layout, class MemSpace>
T norm2(const Vector<T, ExecSpace, Layout, MemSpace>& vec) {
    return Ibis::sqrt(norm2_squared(vec));
}

template <typename T, class ExecSpace, class Layout, class MemSpace>
T norm2_squared(const Vector<T, ExecSpace, Layout, MemSpace>& vec) {
    T norm2;
    Kokkos::parallel_reduce(
        "Vector::norm2", Kokkos::RangePolicy<ExecSpace>(0, vec.size()),
        KOKKOS_LAMBDA(const size_t i, T& utd) {
            T value = vec(i);
            utd += value * value;
        },
        Kokkos::Sum<T>(norm2));
    return norm2;
}

template <typename T, class ExecSpace, class Layout, class MemSpace>
void scale_in_place(Vector<T, ExecSpace, Layout, MemSpace>& vec, const T factor) {
    Kokkos::parallel_for(
        "Ibis::Vector::scale_in_place", Kokkos::RangePolicy<ExecSpace>(0, vec.size()),
        KOKKOS_LAMBDA(const size_t i) { vec(i) *= factor; });
}

template <typename T, class ExecSpace, class Layout1, class Layout2, class MemSpace>
void scale(const Vector<T, ExecSpace, Layout1, MemSpace>& vec,
           Vector<T, ExecSpace, Layout2, MemSpace>& result, const T factor) {
    assert(vec.size() == result.size());
    Kokkos::parallel_for(
        "Ibis::Vector::scale", Kokkos::RangePolicy<ExecSpace>(0, vec.size()),
        KOKKOS_LAMBDA(const size_t i) { result(i) = vec(i) * factor; });
}

template <typename T, class ExecSpace, class Layout1, class Layout2, class MemSpace>
void add_scaled_vector(Vector<T, ExecSpace, Layout1, MemSpace>& vec1,
                       const Vector<T, ExecSpace, Layout2, MemSpace>& vec2, T scale) {
    assert(vec1.size() == vec2.size());
    Kokkos::parallel_for(
        "Ibis::Vector::subtract_scated_vector",
        Kokkos::RangePolicy<ExecSpace>(0, vec1.size()),
        KOKKOS_LAMBDA(const size_t i) { vec1(i) += vec2(i) * scale; });
}

// template <typename T, class ExecSpace, class Layout1, class Layout2, class MemSpace>
// void deep_copy_vector(Vector<T, ExecSpace, Layout1, MemSpace> dest,
//                       const Vector<T, ExecSpace, Layout2, MemSpace>& src) {
// }

template <typename T, class ExecSpace, class MatrixLayout, class VecLayout,
          class ResLayout, class MemSpace>
void gemv(const Matrix<T, ExecSpace, MatrixLayout, MemSpace>& matrix,
          const Vector<T, ExecSpace, VecLayout, MemSpace>& vec,
          Vector<T, ExecSpace, ResLayout, MemSpace>& res) {
    size_t n_rows = matrix.n_rows();
    size_t n_cols = matrix.n_cols();

    assert(vec.size() == n_cols);
    assert(res.size() == n_rows);

    Kokkos::parallel_for(
        "Ibis::gemv", Kokkos::RangePolicy<ExecSpace>(0, n_rows),
        KOKKOS_LAMBDA(const size_t row_i) {
            T dot = T(0.0);
            for (size_t col_i = 0; col_i < n_cols; col_i++) {
                dot += matrix(row_i, col_i) * vec(col_i);
            }
            res(row_i) = dot;
        });
}

template <typename T, class ExecSpace, class LhsLayout, class RhsLayout, class ResLayout,
          class MemSpace>
void gemm(const Matrix<T, ExecSpace, LhsLayout, MemSpace> lhs,
          const Matrix<T, ExecSpace, RhsLayout, MemSpace> rhs,
          Matrix<T, ExecSpace, ResLayout, MemSpace> res) {
    assert(lhs.n_cols() == rhs.n_rows());
    assert(lhs.n_rows() == res.n_rows());
    assert(rhs.n_cols() == res.n_cols());

    // this is probably not a good implementation...
    Kokkos::parallel_for(
        "Ibis::gemm", Kokkos::RangePolicy<ExecSpace>(0, lhs.n_rows()),
        KOKKOS_LAMBDA(const size_t row) {
            for (size_t col = 0; col < rhs.n_cols(); col++) {
                T dot = T(0.0);
                for (size_t i = 0; i < lhs.n_cols(); i++) {
                    dot += lhs(row, i) * rhs(i, col);
                }
                res(row, col) = dot;
            }
        });
}

template <typename T, class ExecSpace, class MatrixLayout, class SolLayout,
          class RhsLayout, class MemSpace>
void upper_triangular_solve(const Matrix<T, ExecSpace, MatrixLayout, MemSpace> A,
                            Vector<T, ExecSpace, SolLayout, MemSpace> sol,
                            Vector<T, ExecSpace, RhsLayout, MemSpace> rhs) {
    assert(A.n_rows() == A.n_cols());
    assert(A.n_rows() == sol.size());
    assert(A.n_rows() == rhs.size());

    int n = A.n_rows();

    sol(n - 1) = rhs(n - 1) / A(n - 1, n - 1);
    for (int i = n - 2; i >= 0; i--) {
        T sum = rhs(i);
        for (int j = i + 1; j < n; j++) {
            sum -= A(i, j) * sol(j);
        }
        sol(i) = sum / A(i, i);
    }
}

template <typename T, class ExecSpace, class Layout1, class Layout2, class MemSpace>
T dot(const Vector<T, ExecSpace, Layout1, MemSpace>& vec1,
      const Vector<T, ExecSpace, Layout2, MemSpace>& vec2) {
    assert(vec1.size() == vec2.size());
    T dot_product;
    Kokkos::parallel_reduce(
        "Ibis::Vector::dot", Kokkos::RangePolicy<ExecSpace>(0, vec1.size()),
        KOKKOS_LAMBDA(const size_t i, T& utd) { utd += vec1(i) * vec2(i); },
        Kokkos::Sum<T>(dot_product));
    return dot_product;
}

}  // namespace Ibis

#endif
