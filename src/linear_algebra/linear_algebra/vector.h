#ifndef LA_VECTOR_H
#define LA_VECTOR_H

#include <util/numeric_types.h>
#include <util/types.h>

#include <Kokkos_Core.hpp>

namespace Ibis {

// template <typename T, class Layout = DefaultArrayLayout, class Space = DefaultMemSpace>
// using Vector = Array1D<T, Layout, Space>;

template <typename T, class Layout = DefaultArrayLayout, class Space = DefaultMemSpace>
class Vector {
public:
    Vector() {}

    Vector(std::string name, size_t n_values) {
        data_ = Array1D<T, Layout, Space> (name, n_values);
    }

    Vector(Array1D<T, Layout, Space> data) : data_(data) {}

    KOKKOS_INLINE_FUNCTION
    T& operator() (const size_t i) { return data_(i); }

    KOKKOS_INLINE_FUNCTION
    T& operator() (const size_t i) const { return data_(i); }

    Vector<T, Kokkos::LayoutStride, Space> sub_vector(const size_t start, const size_t end) {
        return Vector<T, Kokkos::LayoutStride, Space>(
            Kokkos::subview(data_, Kokkos::make_pair(start, end))
        );
    }

    KOKKOS_INLINE_FUNCTION
    size_t size() const { return data_.extent(0); }

private:
    Array1D<T, Layout, Space> data_;
};


template <typename T, class Space = DefaultMemSpace, class Layout = DefaultArrayLayout>
class Matrix {
public:
    Matrix() {}
    
    Matrix(std::string name, const size_t n, const size_t m) {
        data_ = Array2D<T, Layout, Space>(name, n, m);
    }

    Matrix(Array2D<T, Layout, Space> data) : data_(data) {}

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
            Kokkos::subview(data_, Kokkos::make_pair(start_row, end_row),
                                   Kokkos::make_pair(start_col, end_col))
        );
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

    template <typename OtherLayout>
    void deep_copy(Matrix<T, Space, OtherLayout>& other) {
        assert(this->n_rows() == other.n_rows());
        assert(this->n_cols() == other.n_rows());

        size_t n_rows = this->n_rows();
        size_t n_cols = this->n_cols();

        Kokkos::parallel_for("Matrix::deep_copy", n_rows,
                             KOKKOS_LAMBDA(const size_t row){
            for (size_t col = 0; col < n_cols; col++) {
                (*this)(row, col) = other(row, col);
            }                     
        });
    }

    KOKKOS_INLINE_FUNCTION
    size_t n_rows() const { return data_.extent(0); }

    KOKKOS_INLINE_FUNCTION
    size_t n_cols() const { return data_.extent(1); }

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

template <typename T, class Space, class Layout>
void scale_in_place(Vector<T, Layout, Space>& vec, const T factor){
    Kokkos::parallel_for(
        "Ibis::Vector::scale_in_place", vec.size(),
        KOKKOS_LAMBDA(const size_t i) { vec(i) *= factor; });
}

template <typename T, class Space, class Layout1, class Layout2>
void scale(const Vector<T, Layout1, Space>& vec, Vector<T, Layout2, Space>& result, const T factor) {
    assert(vec.size() == result.size());
    Kokkos::parallel_for(
        "Ibis::Vector::scale", vec.size(),
        KOKKOS_LAMBDA(const size_t i) { result(i) = vec(i) * factor; });
}

template <typename T, class Space, class Layout1, class Layout2>
void add_scaled_vector(Vector<T, Layout1, Space>& vec1,
                       const Vector<T, Layout2, Space>& vec2, T scale) {
    assert(vec1.size() == vec2.size());
    Kokkos::parallel_for(
        "Ibis::Vector::subtract_scated_vector", vec1.size(),
        KOKKOS_LAMBDA(const size_t i) { vec1(i) += vec2(i) * scale; });
}

template <typename T, class Space, class Layout1, class Layout2>
void deep_copy_vector(Vector<T, Layout1, Space> dest, const Vector<T, Layout2, Space>& src) {
    assert(dest.size() == src.size());
    // For the moment we'll make this general. If Layout1 and Layout2 are different
    // we cannot use Kokkos::deep_copy. This is the intended use of this function.
    // We could detect if Layout1 and Layout2 are the same, but meh...
    Kokkos::parallel_for("Ibis::deep_copy_vector", dest.size(),
                         KOKKOS_LAMBDA(const size_t i) {
            dest(i) = src(i);            
        }
    );
}

template <typename T, class Space, class MatrixLayout, class VecLayout, class ResLayout>
void gemv(const Matrix<T, Space, MatrixLayout>& matrix,
          const Vector<T, VecLayout, Space>& vec,
          Vector<T, ResLayout, Space>& res) {
    size_t n_rows = matrix.n_rows();
    size_t n_cols = matrix.n_cols();

    assert(vec.size() == n_cols);
    assert(res.size() == n_cols);

    Kokkos::parallel_for("Ibis::gemv", n_rows, KOKKOS_LAMBDA(const size_t row_i){
        T dot = T(0.0);
        for (size_t col_i = 0; col_i < n_cols; col_i++) {
            dot += matrix(row_i, col_i) * vec(col_i);                           
        }
        res(row_i) = dot;
    });
}

template <typename T, class Space, class LhsLayout, class RhsLayout, class ResLayout>
void gemm(const Matrix<T, Space, LhsLayout> lhs, const Matrix<T, Space, RhsLayout> rhs,
          Matrix<T, Space, ResLayout> res) {
    assert(lhs.n_cols() == rhs.n_rows());
    assert(lhs.n_rows() == res.n_rows());
    assert(rhs.n_cols() == res.n_cols());

    // this is probably not a good implementation...
    Kokkos::parallel_for("Ibis::gemm", lhs.n_rows(), KOKKOS_LAMBDA(const size_t row){
        for (size_t col = 0; col < rhs.n_cols(); col++) {
            T dot = T(0.0);
            for (size_t i = 0; i < lhs.n_cols(); i++) {
                dot += lhs(row, i) * rhs(i, col);                          
            }                         
            res(row, col) = dot;
        }
    });
}

template <typename T>
T dot(const Vector<T>& vec1, const Vector<T>& vec2);

}  // namespace Ibis

#endif
