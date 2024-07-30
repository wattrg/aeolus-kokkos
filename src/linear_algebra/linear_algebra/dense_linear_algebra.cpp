#include <linear_algebra/dense_linear_algebra.h>
#include <doctest/doctest.h>

Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace> small_test_vector() {
    Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace> x("x", 4);
    x(0) = 1.0;
    x(1) = 2.0;
    x(2) = -3.0;
    x(3) = 1.5;
    return x;
}

Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace> large_test_vector() {
    Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace> x("x", 8);
    for (size_t i = 0; i < x.size(); i++) {
        x(i) = i;
    }
    return x;
}

TEST_CASE("Ibis::norm2") {
    auto x = small_test_vector();
    CHECK(Ibis::norm2(x) == doctest::Approx(Ibis::sqrt(16.25)));
}

TEST_CASE("Ibis::scale_in_place") {
    auto x = small_test_vector();
    Ibis::scale_in_place(x, 2.0);
    CHECK(x(0) == 2.0);
    CHECK(x(1) == 4.0);
    CHECK(x(2) == -6.0);
    CHECK(x(3) == 3.0);
}

TEST_CASE("Ibis::scale") {
    auto x = small_test_vector();
    Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace> y("y", 4);
    Ibis::scale(x, y, 2.0);
    CHECK(y(0) == 2.0);
    CHECK(y(1) == 4.0);
    CHECK(y(2) == -6.0);
    CHECK(y(3) == 3.0);
}

TEST_CASE("Ibis::add_scaled_vector") {
    auto x = small_test_vector();
    auto y = small_test_vector();
    Ibis::add_scaled_vector(x, y, 2.0);
    CHECK(x(0) == 3.0);    
    CHECK(x(1) == 6.0);    
    CHECK(x(2) == -9.0);    
    CHECK(x(3) == 4.5);    
}

TEST_CASE("Ibis::Vector::sub_vector") {
    auto x = large_test_vector();
    auto x_sub = x.sub_vector(4, 8);
    CHECK(x_sub(0) == 4.0);
    CHECK(x_sub(1) == 5.0);
    CHECK(x_sub(2) == 6.0);
    CHECK(x_sub(3) == 7.0); 
}

TEST_CASE("Ibis::deep_copy_vector") {
    auto x = large_test_vector();
    auto y = small_test_vector();
    auto x_sub = x.sub_vector(0, 4);
    Ibis::deep_copy_vector(x_sub, y);
    CHECK(x(0) == 1.0);
    CHECK(x(1) == 2.0);
    CHECK(x(2) == -3.0);
    CHECK(x(3) == 1.5);
}

TEST_CASE("Ibis::gemm") {
    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> lhs("lhs", 3, 5);
    lhs(0, 0) = 1.0; lhs(0, 1) = 2.0; lhs(0, 2) = 3.0; lhs(0, 3) = 4.0; lhs(0, 4) = 5.0;
    lhs(1, 0) = 6.0; lhs(1, 1) = 7.0; lhs(1, 2) = 8.0; lhs(1, 3) = 9.0; lhs(1, 4) = 10.0;
    lhs(2, 0) = 11.0; lhs(2, 1) = 12.0; lhs(2, 2) = 13.0; lhs(2, 3) = 14.0; lhs(2, 4) = 15.0;
    
    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> rhs("lhs", 5, 2);
    rhs(0, 0) = 1.0; rhs(0, 1) = 2.0;
    rhs(1, 0) = 3.0; rhs(1, 1) = 4.0;
    rhs(2, 0) = 5.0; rhs(2, 1) = 6.0;
    rhs(3, 0) = 7.0; rhs(3, 1) = 8.0;
    rhs(4, 0) = 9.0; rhs(4, 1) = 10.0;

    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> res("res", 3, 2);

    Ibis::gemm(lhs, rhs, res);

    CHECK(res(0, 0) == 95.0); CHECK(res(0, 1) == 110.0);
    CHECK(res(1, 0) == 220.0); CHECK(res(1, 1) == 260.0);
    CHECK(res(2, 0) == 345.0); CHECK(res(2, 1) == 410.0);
}

TEST_CASE("Ibis::gemv") {
    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> lhs("lhs", 3, 5);
    lhs(0, 0) = 1.0; lhs(0, 1) = 2.0; lhs(0, 2) = 3.0; lhs(0, 3) = 4.0; lhs(0, 4) = 5.0;
    lhs(1, 0) = 6.0; lhs(1, 1) = 7.0; lhs(1, 2) = 8.0; lhs(1, 3) = 9.0; lhs(1, 4) = 10.0;
    lhs(2, 0) = 11.0; lhs(2, 1) = 12.0; lhs(2, 2) = 13.0; lhs(2, 3) = 14.0; lhs(2, 4) = 15.0;

    Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace> vec("vec", 5);
    vec(0) = 1.0;
    vec(1) = 2.0;
    vec(2) = 3.0;
    vec(3) = 4.0;
    vec(4) = 5.0;

    Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace> res("res", 3);

    Ibis::gemv(lhs, vec, res);
    CHECK(res(0) == 55.0);
    CHECK(res(1) == 130.0);
    CHECK(res(2) == 205.0);
}

TEST_CASE("Ibis::dot") {
    Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace> a("a", 3);
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;

    Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace> b("b", 3);
    b(0) = 2.0;
    b(1) = 3.0;
    b(2) = 4.0;
    
    Ibis::real dot = Ibis::dot(a, b);

    CHECK(dot == 20.0);
}

TEST_CASE("Ibis::Matrix::column") {
    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> A("A", 3, 5);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0; A(0, 3) = 4.0; A(0, 4) = 5.0;
    A(1, 0) = 6.0; A(1, 1) = 7.0; A(1, 2) = 8.0; A(1, 3) = 9.0; A(1, 4) = 10.0;
    A(2, 0) = 11.0; A(2, 1) = 12.0; A(2, 2) = 13.0; A(2, 3) = 14.0; A(2, 4) = 15.0;
    
    auto col = A.column(0);

    CHECK(col(0) == 1.0);
    CHECK(col(1) == 6.0);
    CHECK(col(2) == 11.0);
}

TEST_CASE("Ibis::Matrix::column") {
    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> A("A", 3, 5);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0; A(0, 3) = 4.0; A(0, 4) = 5.0;
    A(1, 0) = 6.0; A(1, 1) = 7.0; A(1, 2) = 8.0; A(1, 3) = 9.0; A(1, 4) = 10.0;
    A(2, 0) = 11.0; A(2, 1) = 12.0; A(2, 2) = 13.0; A(2, 3) = 14.0; A(2, 4) = 15.0;
    
    auto row = A.row(1);

    CHECK(row(0) == 6.0);
    CHECK(row(1) == 7.0);
    CHECK(row(2) == 8.0);
    CHECK(row(3) == 9.0);
    CHECK(row(4) == 10.0);
}

TEST_CASE("Ibis::Matrix::sub_matrix") {
    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> A("A", 3, 5);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0; A(0, 3) = 4.0; A(0, 4) = 5.0;
    A(1, 0) = 6.0; A(1, 1) = 7.0; A(1, 2) = 8.0; A(1, 3) = 9.0; A(1, 4) = 10.0;
    A(2, 0) = 11.0; A(2, 1) = 12.0; A(2, 2) = 13.0; A(2, 3) = 14.0; A(2, 4) = 15.0;

    auto A_sub = A.sub_matrix(0, 2, 1, 3);

    CHECK(A_sub(0, 0) == 2.0); CHECK(A_sub(0, 1) == 3.0);
    CHECK(A_sub(1, 0) == 7.0); CHECK(A_sub(1, 1) == 8.0);
}

TEST_CASE("Ibis::Matrix::deep_copy") {
    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> A("A", 3, 3);
    A.set_to_identity();
    auto A_sub = A.sub_matrix(0, 2, 0, 2);

    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> B("B", 2, 2);
    B(0, 0) = 2.0;
    B(1, 1) = 3.0;
    A_sub.deep_copy(B);

    CHECK(A(0, 0) == 2.0); CHECK(A(0, 1) == 0.0); CHECK(A(0, 2) == 0.0);
    CHECK(A(1, 0) == 0.0); CHECK(A(1, 1) == 3.0); CHECK(A(1, 2) == 0.0);
    CHECK(A(2, 0) == 0.0); CHECK(A(2, 1) == 0.0); CHECK(A(2, 2) == 1.0);
}

TEST_CASE("Ibis::upper_triangular_solve") {
    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> A("A", 3, 3);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
                   A(1, 1) = 5.0; A(1, 2) = 2.0;
                                  A(2, 2) = 3.0;

    Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace> rhs("rhs", 3);
    rhs(0) = 1.0; rhs(1) = 2.0; rhs(2) = 3.0;
    
    Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace> sol("sol", 3);

    Ibis::upper_triangular_solve(A, sol, rhs);

    CHECK(sol(0) == -2.0);
    CHECK(sol(1) == 0.0);
    CHECK(sol(2) == 1.0);
}

TEST_CASE("Ibis::Matrix::columns") {
    Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace> lhs("lhs", 3, 5);
    lhs(0, 0) = 1.0; lhs(0, 1) = 2.0; lhs(0, 2) = 3.0; lhs(0, 3) = 4.0; lhs(0, 4) = 5.0;
    lhs(1, 0) = 6.0; lhs(1, 1) = 7.0; lhs(1, 2) = 8.0; lhs(1, 3) = 9.0; lhs(1, 4) = 10.0;
    lhs(2, 0) = 11.0; lhs(2, 1) = 12.0; lhs(2, 2) = 13.0; lhs(2, 3) = 14.0; lhs(2, 4) = 15.0;

    auto columns = lhs.columns(0, 2);
    CHECK(columns.n_rows() == 3);
    CHECK(columns.n_cols() == 2);
    CHECK(columns(0, 0) == 1.0);
}
