#include <doctest/doctest.h>
#include <linear_algebra/dense_linear_algebra.h>
#include "Kokkos_Core_fwd.hpp"

Ibis::Vector<Ibis::real> small_test_vector() {
    Ibis::Vector<Ibis::real> x("x", 4);
    auto x_h = x.host_mirror();
    x_h(0) = 1.0;
    x_h(1) = 2.0;
    x_h(2) = -3.0;
    x_h(3) = 1.5;
    x.deep_copy_space(x_h);
    return x;
}

Ibis::Vector<Ibis::real> large_test_vector() {
    Ibis::Vector<Ibis::real> x("x", 8);
    auto x_h = x.host_mirror();
    for (size_t i = 0; i < x_h.size(); i++) {
        x_h(i) = i;
    }
    x.deep_copy_space(x_h);
    return x;
}

TEST_CASE("Ibis::norm2") {
    auto x = small_test_vector();
    CHECK(Ibis::norm2(x) == doctest::Approx(Ibis::sqrt(16.25)));
}

TEST_CASE("Ibis::scale_in_place") {
    auto x = small_test_vector();
    Ibis::scale_in_place(x, 2.0);
    auto x_h = x.host_mirror();
    x_h.deep_copy_space(x);
    CHECK(x_h(0) == 2.0);
    CHECK(x_h(1) == 4.0);
    CHECK(x_h(2) == -6.0);
    CHECK(x_h(3) == 3.0);
}

TEST_CASE("Ibis::scale") {
    auto x = small_test_vector();
    Ibis::Vector<Ibis::real> y("y", 4);
    Ibis::scale(x, y, 2.0);
    auto y_h = y.host_mirror();
    y_h.deep_copy_space(y);
    CHECK(y_h(0) == 2.0);
    CHECK(y_h(1) == 4.0);
    CHECK(y_h(2) == -6.0);
    CHECK(y_h(3) == 3.0);
}

TEST_CASE("Ibis::add_scaled_vector") {
    auto x = small_test_vector();
    auto y = small_test_vector();
    Ibis::add_scaled_vector(x, y, 2.0);
    auto x_h = x.host_mirror();
    x_h.deep_copy_space(x);
    CHECK(x_h(0) == 3.0);
    CHECK(x_h(1) == 6.0);
    CHECK(x_h(2) == -9.0);
    CHECK(x_h(3) == 4.5);
}

TEST_CASE("Ibis::Vector::sub_vector") {
    auto x = large_test_vector();
    auto x_sub = x.sub_vector(4, 8);
    auto x_sub_h = x_sub.host_mirror();
    x_sub_h.deep_copy_space(x_sub);
    CHECK(x_sub_h(0) == 4.0);
    CHECK(x_sub_h(1) == 5.0);
    CHECK(x_sub_h(2) == 6.0);
    CHECK(x_sub_h(3) == 7.0);
}

TEST_CASE("Ibis::Vector::deep_copy_layout") {
    auto x = large_test_vector();
    auto y = small_test_vector();
    auto x_sub = x.sub_vector(0, 4);
    x_sub.deep_copy_layout(y);
    auto x_h = x.host_mirror();
    x_h.deep_copy_space(x);
    CHECK(x_h(0) == 1.0);
    CHECK(x_h(1) == 2.0);
    CHECK(x_h(2) == -3.0);
    CHECK(x_h(3) == 1.5);
}

TEST_CASE("Ibis::gemm") {
    Ibis::Matrix<Ibis::real> lhs("lhs", 3, 5);
    auto lhs_h = lhs.host_mirror();
    lhs_h(0, 0) = 1.0;
    lhs_h(0, 1) = 2.0;
    lhs_h(0, 2) = 3.0;
    lhs_h(0, 3) = 4.0;
    lhs_h(0, 4) = 5.0;
    lhs_h(1, 0) = 6.0;
    lhs_h(1, 1) = 7.0;
    lhs_h(1, 2) = 8.0;
    lhs_h(1, 3) = 9.0;
    lhs_h(1, 4) = 10.0;
    lhs_h(2, 0) = 11.0;
    lhs_h(2, 1) = 12.0;
    lhs_h(2, 2) = 13.0;
    lhs_h(2, 3) = 14.0;
    lhs_h(2, 4) = 15.0;
    lhs.deep_copy_space(lhs_h);

    Ibis::Matrix<Ibis::real> rhs("lhs", 5, 2);
    auto rhs_h = rhs.host_mirror();
    rhs_h(0, 0) = 1.0;
    rhs_h(0, 1) = 2.0;
    rhs_h(1, 0) = 3.0;
    rhs_h(1, 1) = 4.0;
    rhs_h(2, 0) = 5.0;
    rhs_h(2, 1) = 6.0;
    rhs_h(3, 0) = 7.0;
    rhs_h(3, 1) = 8.0;
    rhs_h(4, 0) = 9.0;
    rhs_h(4, 1) = 10.0;
    rhs.deep_copy_space(rhs_h);

    Ibis::Matrix<Ibis::real> res("res", 3, 2);
    auto res_h = res.host_mirror();

    Ibis::gemm(lhs, rhs, res);

    res_h.deep_copy_space(res);
    CHECK(res_h(0, 0) == 95.0);
    CHECK(res_h(0, 1) == 110.0);
    CHECK(res_h(1, 0) == 220.0);
    CHECK(res_h(1, 1) == 260.0);
    CHECK(res_h(2, 0) == 345.0);
    CHECK(res_h(2, 1) == 410.0);
}

TEST_CASE("Ibis::gemv") {
    Ibis::Matrix<Ibis::real> lhs("lhs", 3, 5);
    auto lhs_h = lhs.host_mirror();
    lhs_h(0, 0) = 1.0;
    lhs_h(0, 1) = 2.0;
    lhs_h(0, 2) = 3.0;
    lhs_h(0, 3) = 4.0;
    lhs_h(0, 4) = 5.0;
    lhs_h(1, 0) = 6.0;
    lhs_h(1, 1) = 7.0;
    lhs_h(1, 2) = 8.0;
    lhs_h(1, 3) = 9.0;
    lhs_h(1, 4) = 10.0;
    lhs_h(2, 0) = 11.0;
    lhs_h(2, 1) = 12.0;
    lhs_h(2, 2) = 13.0;
    lhs_h(2, 3) = 14.0;
    lhs_h(2, 4) = 15.0;
    lhs.deep_copy_space(lhs_h);

    Ibis::Vector<Ibis::real> vec("vec", 5);
    auto vec_h = vec.host_mirror();
    vec_h(0) = 1.0;
    vec_h(1) = 2.0;
    vec_h(2) = 3.0;
    vec_h(3) = 4.0;
    vec_h(4) = 5.0;
    vec.deep_copy_space(vec_h);

    Ibis::Vector<Ibis::real> res("res", 3);
    auto res_h = res.host_mirror();

    Ibis::gemv(lhs, vec, res);
    res_h.deep_copy_space(res);
    CHECK(res_h(0) == 55.0);
    CHECK(res_h(1) == 130.0);
    CHECK(res_h(2) == 205.0);
}

TEST_CASE("Ibis::dot") {
    Ibis::Vector<Ibis::real> a("a", 3);
    auto a_h = a.host_mirror();
    a_h(0) = 1.0;
    a_h(1) = 2.0;
    a_h(2) = 3.0;
    a.deep_copy_space(a_h);

    Ibis::Vector<Ibis::real> b("b", 3);
    auto b_h = b.host_mirror();
    b_h(0) = 2.0;
    b_h(1) = 3.0;
    b_h(2) = 4.0;
    b.deep_copy_space(b_h);

    Ibis::real dot = Ibis::dot(a, b);

    CHECK(dot == 20.0);
}

TEST_CASE("Ibis::Matrix::column") {
    Ibis::Matrix<Ibis::real> A("A", 3, 5);
    auto A_h = A.host_mirror();
    A_h(0, 0) = 1.0;
    A_h(0, 1) = 2.0;
    A_h(0, 2) = 3.0;
    A_h(0, 3) = 4.0;
    A_h(0, 4) = 5.0;
    A_h(1, 0) = 6.0;
    A_h(1, 1) = 7.0;
    A_h(1, 2) = 8.0;
    A_h(1, 3) = 9.0;
    A_h(1, 4) = 10.0;
    A_h(2, 0) = 11.0;
    A_h(2, 1) = 12.0;
    A_h(2, 2) = 13.0;
    A_h(2, 3) = 14.0;
    A_h(2, 4) = 15.0;
    A.deep_copy_space(A_h);

    auto col = A.column(0);
    auto col_h = col.host_mirror();
    col_h.deep_copy_space(col);

    CHECK(col_h(0) == 1.0);
    CHECK(col_h(1) == 6.0);
    CHECK(col_h(2) == 11.0);
}

TEST_CASE("Ibis::Matrix::row") {
    // we can't copy non-contiguous data between devices,
    // so this will be CPU only
    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> A_h("A", 3, 5);
    A_h(0, 0) = 1.0;
    A_h(0, 1) = 2.0;
    A_h(0, 2) = 3.0;
    A_h(0, 3) = 4.0;
    A_h(0, 4) = 5.0;
    A_h(1, 0) = 6.0;
    A_h(1, 1) = 7.0;
    A_h(1, 2) = 8.0;
    A_h(1, 3) = 9.0;
    A_h(1, 4) = 10.0;
    A_h(2, 0) = 11.0;
    A_h(2, 1) = 12.0;
    A_h(2, 2) = 13.0;
    A_h(2, 3) = 14.0;
    A_h(2, 4) = 15.0;


    auto row = A_h.row(1);

    CHECK(row(0) == 6.0);
    CHECK(row(1) == 7.0);
    CHECK(row(2) == 8.0);
    CHECK(row(3) == 9.0);
    CHECK(row(4) == 10.0);
}

TEST_CASE("Ibis::Matrix::sub_matrix") {
    // we can't copy non-contiguous data between devices,
    // so this will be CPU only
    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> A_h("A", 3, 5);
    A_h(0, 0) = 1.0;
    A_h(0, 1) = 2.0;
    A_h(0, 2) = 3.0;
    A_h(0, 3) = 4.0;
    A_h(0, 4) = 5.0;
    A_h(1, 0) = 6.0;
    A_h(1, 1) = 7.0;
    A_h(1, 2) = 8.0;
    A_h(1, 3) = 9.0;
    A_h(1, 4) = 10.0;
    A_h(2, 0) = 11.0;
    A_h(2, 1) = 12.0;
    A_h(2, 2) = 13.0;
    A_h(2, 3) = 14.0;
    A_h(2, 4) = 15.0;

    auto A_sub = A_h.sub_matrix(0, 2, 1, 3);

    CHECK(A_sub(0, 0) == 2.0);
    CHECK(A_sub(0, 1) == 3.0);
    CHECK(A_sub(1, 0) == 7.0);
    CHECK(A_sub(1, 1) == 8.0);
}

TEST_CASE("Ibis::Matrix::deep_copy") {
    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> A("A", 3, 3);
    A.set_to_identity();
    auto A_sub = A.sub_matrix(0, 2, 0, 2);

    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> B("B", 2, 2);
    B(0, 0) = 2.0;
    B(1, 1) = 3.0;
    A_sub.deep_copy(B);

    CHECK(A(0, 0) == 2.0);
    CHECK(A(0, 1) == 0.0);
    CHECK(A(0, 2) == 0.0);
    CHECK(A(1, 0) == 0.0);
    CHECK(A(1, 1) == 3.0);
    CHECK(A(1, 2) == 0.0);
    CHECK(A(2, 0) == 0.0);
    CHECK(A(2, 1) == 0.0);
    CHECK(A(2, 2) == 1.0);
}

TEST_CASE("Ibis::upper_triangular_solve") {
    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> A("A", 3, 3);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;
    A(1, 1) = 5.0;
    A(1, 2) = 2.0;
    A(2, 2) = 3.0;

    Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace> rhs("rhs", 3);
    rhs(0) = 1.0;
    rhs(1) = 2.0;
    rhs(2) = 3.0;

    Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace> sol("sol", 3);

    Ibis::upper_triangular_solve(A, sol, rhs);

    CHECK(sol(0) == -2.0);
    CHECK(sol(1) == 0.0);
    CHECK(sol(2) == 1.0);
}

TEST_CASE("Ibis::Matrix::columns") {
    Ibis::Matrix<Ibis::real> lhs("lhs", 3, 5);
    auto lhs_h = lhs.host_mirror();
    lhs_h(0, 0) = 1.0;
    lhs_h(0, 1) = 2.0;
    lhs_h(0, 2) = 3.0;
    lhs_h(0, 3) = 4.0;
    lhs_h(0, 4) = 5.0;
    lhs_h(1, 0) = 6.0;
    lhs_h(1, 1) = 7.0;
    lhs_h(1, 2) = 8.0;
    lhs_h(1, 3) = 9.0;
    lhs_h(1, 4) = 10.0;
    lhs_h(2, 0) = 11.0;
    lhs_h(2, 1) = 12.0;
    lhs_h(2, 2) = 13.0;
    lhs_h(2, 3) = 14.0;
    lhs_h(2, 4) = 15.0;
    lhs.deep_copy_space(lhs_h);

    auto columns = lhs.columns(0, 2);
    CHECK(columns.n_rows() == 3);
    CHECK(columns.n_cols() == 2);
    // CHECK(columns(0, 0) == 1.0);
}
