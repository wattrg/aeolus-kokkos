#include "vector3.h"

#include <doctest/doctest.h>

#include <Kokkos_MathematicalFunctions.hpp>
#include <cassert>
#include <cmath>

#define VEC3_TOL 1e-15

template <typename T>
void dot(const Vector3s<T> &a, const Vector3s<T> &b, Field<T> &result) {
    assert((a.size() == b.size()) && (b.size() == result.size()));

    Kokkos::parallel_for(
        "vector3 dot product", a.size(), KOKKOS_LAMBDA(const size_t i) {
            result(i) = a(i, 0) * b(i, 0) + a(i, 1) * b(i, 1) + a(i, 2) * b(i, 2);
        });
}
template void dot<double>(const Vector3s<double> &a, const Vector3s<double> &b,
                          Field<double> &result);

template <typename T>
void add(const Vector3s<T> &a, const Vector3s<T> &b, Vector3s<T> &result) {
    assert((a.size() == b.size()) && (b.size() == result.size()));

    Kokkos::parallel_for(
        "Vector3s add", a.size(), KOKKOS_LAMBDA(const size_t i) {
            result(i, 0) = a(i, 0) + b(i, 0);
            result(i, 1) = a(i, 1) + b(i, 1);
            result(i, 2) = a(i, 2) + b(i, 2);
        });
}
template void add<Ibis::real>(const Vector3s<Ibis::real> &a,
                              const Vector3s<Ibis::real> &b,
                              Vector3s<Ibis::real> &result);

template <typename T>
void subtract(const Vector3s<T> &a, const Vector3s<T> &b, Vector3s<T> &result) {
    assert((a.size() == b.size()) && (b.size() == result.size()));

    Kokkos::parallel_for(
        "Vector3s subtract", a.size(), KOKKOS_LAMBDA(const size_t i) {
            result(i, 0) = a(i, 0) - b(i, 0);
            result(i, 1) = a(i, 1) - b(i, 1);
            result(i, 2) = a(i, 2) - b(i, 2);
        });
}
template void subtract<Ibis::real>(const Vector3s<Ibis::real> &a,
                                   const Vector3s<Ibis::real> &b,
                                   Vector3s<Ibis::real> &result);

template <typename T>
void cross(const Vector3s<T> &a, const Vector3s<T> &b, Vector3s<T> &result) {
    assert((a.size() == b.size()) && (b.size() == result.size()));

    Kokkos::parallel_for(
        "Vector3s cross", a.size(), KOKKOS_LAMBDA(const size_t i) {
            result(i, 0) = a(i, 1) * b(i, 2) - a(i, 2) * b(i, 1);
            result(i, 1) = a(i, 2) * b(i, 0) - a(i, 0) * b(i, 2);
            result(i, 2) = a(i, 0) * b(i, 1) - a(i, 1) * b(i, 0);
        });
}
template void cross<double>(const Vector3s<Ibis::real> &a, const Vector3s<Ibis::real> &b,
                            Vector3s<Ibis::real> &result);

template <typename T>
void scale_in_place(Vector3s<T> &a, T factor) {
    Kokkos::parallel_for(
        "Vector3s scale", a.size(), KOKKOS_LAMBDA(const size_t i) {
            a(i, 0) *= factor;
            a(i, 1) *= factor;
            a(i, 2) *= factor;
        });
}
template void scale_in_place<double>(Vector3s<Ibis::real> &a, Ibis::real factor);

template <typename T>
void length(const Vector3s<T> &a, Field<T> &len) {
    assert(a.size() == len.size());

    Kokkos::parallel_for(
        "Vector3s length", a.size(), KOKKOS_LAMBDA(const size_t i) {
            len(i) =
                Ibis::sqrt(a(i, 0) * a(i, 0) + a(i, 1) * a(i, 1) + a(i, 2) * a(i, 2));
        });
}
template void length<Ibis::real>(const Vector3s<Ibis::real> &a, Field<Ibis::real> &len);

template <typename T>
void normalise(Vector3s<T> &a) {
    Kokkos::parallel_for(
        "Vector3s normalise", a.size(), KOKKOS_LAMBDA(const size_t i) {
            double length_inv = 1. / Ibis::sqrt(a(i, 0) * a(i, 0) + a(i, 1) * a(i, 1) +
                                                a(i, 2) * a(i, 2));
            a(i, 0) *= length_inv;
            a(i, 1) *= length_inv;
            a(i, 2) *= length_inv;
        });
}
template void normalise<double>(Vector3s<Ibis::real> &a);

template <typename T>
void transform_to_local_frame(Vector3s<T> &a, const Vector3s<T> &norm,
                              const Vector3s<T> &tan1, const Vector3s<T> tan2) {
    Kokkos::parallel_for(
        "Vector3s::transform_to_local_frame", a.size(), KOKKOS_LAMBDA(const size_t i) {
            T x = a.x(i) * norm.x(i) + a.y(i) * norm.y(i) + a.z(i) * norm.z(i);
            T y = a.x(i) * tan1.x(i) + a.y(i) * tan1.y(i) + a.z(i) * tan1.z(i);
            T z = a.x(i) * tan2.x(i) + a.y(i) * tan2.y(i) + a.z(i) * tan2.z(i);
            a.x(i) = x;
            a.y(i) = y;
            a.z(i) = z;
        });
}
template void transform_to_local_frame<Ibis::real>(Vector3s<Ibis::real> &a,
                                                   const Vector3s<Ibis::real> &norm,
                                                   const Vector3s<Ibis::real> &tan1,
                                                   const Vector3s<Ibis::real> tan2);
template void transform_to_local_frame<Ibis::dual>(Vector3s<Ibis::dual> &a,
                                                   const Vector3s<Ibis::dual> &norm,
                                                   const Vector3s<Ibis::dual> &tan1,
                                                   const Vector3s<Ibis::dual> tan2);

template <typename T>
void transform_to_global_frame(Vector3s<T> &a, const Vector3s<T> &norm,
                               const Vector3s<T> &tan1, const Vector3s<T> &tan2) {
    Kokkos::parallel_for(
        "Vector3s::transform_to_global_frame", a.size(), KOKKOS_LAMBDA(const size_t i) {
            T x = a(i, 0) * norm(i, 0) + a(i, 1) * tan1(i, 0) + a(i, 2) * tan2(i, 0);
            T y = a(i, 0) * norm(i, 1) + a(i, 1) * tan1(i, 1) + a(i, 2) * tan2(i, 1);
            T z = a(i, 0) * norm(i, 2) + a(i, 1) * tan1(i, 2) + a(i, 2) * tan2(i, 2);
            a.x(i) = x;
            a.y(i) = y;
            a.z(i) = z;
        });
}

template void transform_to_global_frame<Ibis::real>(Vector3s<Ibis::real> &a,
                                                    const Vector3s<Ibis::real> &norm,
                                                    const Vector3s<Ibis::real> &tan1,
                                                    const Vector3s<Ibis::real> &tan2);
template void transform_to_global_frame<Ibis::dual>(Vector3s<Ibis::dual> &a,
                                                    const Vector3s<Ibis::dual> &norm,
                                                    const Vector3s<Ibis::dual> &tan1,
                                                    const Vector3s<Ibis::dual> &tan2);

TEST_CASE("Vector Dot Product") {
    size_t n = 10;

    // device memory
    Vector3s<double> a_dev("a", n);
    Vector3s<double> b_dev("b", n);
    Field<double> result_dev("result", n);

    // host memory
    auto a_host = a_dev.host_mirror();
    auto b_host = b_dev.host_mirror();
    auto result_host = result_dev.host_mirror();
    Field<double>::mirror_type expected("expected", n);

    // set some data
    for (size_t i = 0; i < n; i++) {
        a_host(i, 0) = 1.0 * i;
        a_host(i, 1) = 2.0 * i;
        a_host(i, 2) = 3.0 * i;

        b_host(i, 0) = 1.0 * i * i;
        b_host(i, 1) = 2.0 * i * i;
        b_host(i, 2) = 3.0 * i * i;

        expected(i) = a_host.x(i) * b_host.x(i) + a_host.y(i) * b_host.y(i) +
                      a_host.z(i) * b_host.z(i);
    }

    // copy data to the device
    a_dev.deep_copy(a_host);
    b_dev.deep_copy(b_host);
    result_dev.deep_copy(result_host);

    // do the work
    dot(a_dev, b_dev, result_dev);

    // copy result back
    result_host.deep_copy(result_dev);

    // check the results are correct
    for (size_t i = 0; i < n; i++) {
        CHECK(Kokkos::fabs(expected(i) < result_host(i)) < VEC3_TOL);
    }
}

TEST_CASE("Vector3s Add") {
    size_t n = 20;

    // allocate device memory
    Vector3s<double> a_dev("a", n);
    Vector3s<double> b_dev("b", n);
    Vector3s<double> result_dev("result", n);

    // allocate host memory
    auto a_host = a_dev.host_mirror();
    auto b_host = b_dev.host_mirror();
    auto result_host = result_dev.host_mirror();
    Vector3s<double>::mirror_type expected("expected", n);

    // set some data
    for (size_t i = 0; i < n; i++) {
        a_host(i, 0) = 1.0 * i;
        a_host(i, 1) = -2.0 * i;
        a_host(i, 2) = 3.0 * i - 5;

        b_host(i, 0) = -1.0 * i * i;
        b_host(i, 1) = 0.5 * (i - 1) * i;
        b_host(i, 2) = 3.0 * i * i;

        expected(i, 0) = a_host(i, 0) + b_host(i, 0);
        expected(i, 1) = a_host(i, 1) + b_host(i, 1);
        expected(i, 2) = a_host(i, 2) + b_host(i, 2);
    }

    // copy data to the device
    a_dev.deep_copy(a_host);
    b_dev.deep_copy(b_host);

    // do the work
    add(a_dev, b_dev, result_dev);

    // copy the result to the host
    result_host.deep_copy(result_dev);

    // check the results are correct
    for (size_t i = 0; i < n; i++) {
        CHECK(Kokkos::fabs(expected(i, 0) - result_host(i, 0)) < VEC3_TOL);
        CHECK(Kokkos::fabs(expected(i, 1) - result_host(i, 1)) < VEC3_TOL);
        CHECK(Kokkos::fabs(expected(i, 2) - result_host(i, 2)) < VEC3_TOL);
    }
}

TEST_CASE("Vector3s subtract") {
    size_t n = 20;
    // allocate memory on the device
    Vector3s<double> a_dev("a", n);
    Vector3s<double> b_dev("b", n);
    Vector3s<double> result_dev("result", n);

    // allocate memory on the host
    auto a_host = a_dev.host_mirror();
    auto b_host = b_dev.host_mirror();
    auto result_host = result_dev.host_mirror();
    Vector3s<double>::mirror_type expected("expected", n);

    // set some data
    for (size_t i = 0; i < n; i++) {
        a_host(i, 0) = 1.0 * i;
        a_host(i, 1) = -2.0 * i;
        a_host(i, 2) = 3.0 * i - 5;

        b_host(i, 0) = -1.0 * i * i;
        b_host(i, 1) = 0.5 * (i - 1) * i;
        b_host(i, 2) = 3.0 * i * i;

        expected(i, 0) = a_host(i, 0) - b_host(i, 0);
        expected(i, 1) = a_host(i, 1) - b_host(i, 1);
        expected(i, 2) = a_host(i, 2) - b_host(i, 2);
    }

    // copy data to the device
    a_dev.deep_copy(a_host);
    b_dev.deep_copy(b_host);

    // do the work
    subtract(a_dev, b_dev, result_dev);

    // copy result to the host
    result_host.deep_copy(result_dev);

    // check the results are correct
    for (size_t i = 0; i < n; i++) {
        CHECK(Kokkos::fabs(expected(i, 0) - result_host(i, 0)) < VEC3_TOL);
        CHECK(Kokkos::fabs(expected(i, 1) - result_host(i, 1)) < VEC3_TOL);
        CHECK(Kokkos::fabs(expected(i, 2) - result_host(i, 2)) < VEC3_TOL);
    }
}

TEST_CASE("Vector3s cross") {
    size_t n = 20;

    // allocate memory on the device
    Vector3s<double> a_dev("a", n);
    Vector3s<double> b_dev("b", n);
    Vector3s<double> result_dev("result", n);

    // allocate memory on the host
    auto a_host = a_dev.host_mirror();
    auto b_host = b_dev.host_mirror();
    auto result_host = result_dev.host_mirror();
    Vector3s<double>::mirror_type expected("expected", n);

    // set some data
    for (size_t i = 0; i < n; i++) {
        a_host(i, 0) = 1.0 * i;
        a_host(i, 1) = -2.0 * i;
        a_host(i, 2) = 3.0 * i - 5;

        b_host(i, 0) = -1.0 * i * i;
        b_host(i, 1) = 0.5 * (i - 1) * i;
        b_host(i, 2) = 3.0 * i * i;

        expected.x(i) = a_host.y(i) * b_host.z(i) - a_host.z(i) * b_host.y(i);
        expected.y(i) = a_host.z(i) * b_host.x(i) - a_host.x(i) * b_host.z(i);
        expected.z(i) = a_host.x(i) * b_host.y(i) - a_host.y(i) * b_host.x(i);
    }

    // copy data to the device
    a_dev.deep_copy(a_host);
    b_dev.deep_copy(b_host);

    // do the work
    cross(a_dev, b_dev, result_dev);

    // copy result to the host
    result_host.deep_copy(result_dev);

    // check the results are correct
    for (size_t i = 0; i < n; i++) {
        CHECK(Kokkos::fabs(expected(i, 0) - result_host(i, 0)) < VEC3_TOL);
        CHECK(Kokkos::fabs(expected(i, 1) - result_host(i, 1)) < VEC3_TOL);
        CHECK(Kokkos::fabs(expected(i, 2) - result_host(i, 2)) < VEC3_TOL);
    }
}

TEST_CASE("Vector3s scale_in_place") {
    size_t n = 20;
    Vector3s<double> a_dev("a", n);
    auto a_host = a_dev.host_mirror();
    double factor = 2.0;

    for (size_t i = 0; i < n; i++) {
        a_host(i, 0) = 1.0 * i;
        a_host(i, 1) = 2.0 * i;
        a_host(i, 2) = 3.0 * i;
    }

    a_dev.deep_copy(a_host);
    scale_in_place(a_dev, factor);
    a_host.deep_copy(a_dev);

    for (size_t i = 0; i < n; i++) {
        CHECK(Kokkos::fabs(a_host.x(i) - 2.0 * i) < VEC3_TOL);
        CHECK(Kokkos::fabs(a_host.y(i) - 4.0 * i) < VEC3_TOL);
        CHECK(Kokkos::fabs(a_host.z(i) - 6.0 * i) < VEC3_TOL);
    }
}

TEST_CASE("Vector3s length") {
    size_t n = 20;
    Vector3s<double> a_dev("a", n);
    Field<double> len_dev("length", n);

    auto a_host = a_dev.host_mirror();
    auto len_host = len_dev.host_mirror();
    Field<double>::mirror_type expected("expected", n);

    for (size_t i = 0; i < n; i++) {
        a_host(i, 0) = 1.0 * i;
        a_host(i, 1) = 2.0 * i;
        a_host(i, 2) = 3.0 * i;

        expected(i) = Ibis::sqrt(a_host.x(i) * a_host.x(i) + a_host.y(i) * a_host.y(i) +
                                 a_host.z(i) * a_host.z(i));
    }

    a_dev.deep_copy(a_host);
    length(a_dev, len_dev);
    len_host.deep_copy(len_dev);

    for (size_t i = 0; i < n; i++) {
        CHECK(Kokkos::fabs(len_host(i) - expected(i)) < VEC3_TOL);
    }
}

TEST_CASE("Vector3s normalise") {
    size_t n = 20;
    Vector3s<double> a_dev("a", n);
    auto a_host = a_dev.host_mirror();

    for (size_t i = 0; i < n; i++) {
        a_host.x(i) = 1.0 * i + 1.0;
        a_host.y(i) = 2.0 * i;
        a_host.z(i) = 3.0 * i;
    }

    a_dev.deep_copy(a_host);
    normalise(a_dev);
    a_host.deep_copy(a_dev);

    for (size_t i = 0; i < n; i++) {
        double length_inv = Ibis::sqrt((i + 1.0) * (i + 1.0) + 4.0 * i * i + 9.0 * i * i);
        CHECK(Kokkos::fabs(a_host.x(i) - 1.0 * (i + 1.0) / length_inv) < VEC3_TOL);
        CHECK(Kokkos::fabs(a_host.y(i) - 2.0 * i / length_inv) < VEC3_TOL);
        CHECK(Kokkos::fabs(a_host.z(i) - 3.0 * i / length_inv) < VEC3_TOL);
    }
}

TEST_CASE("Vector3s::transform_to_local_frame") {
    size_t n = 3;
    Vector3s<double> a_dev("a", n);
    Vector3s<double> norm_dev("norm", n);
    Vector3s<double> tan1_dev("tan1", n);
    Vector3s<double> tan2_dev("tan2", n);

    auto a_host = a_dev.host_mirror();
    auto norm_host = norm_dev.host_mirror();
    auto tan1_host = tan1_dev.host_mirror();
    auto tan2_host = tan2_dev.host_mirror();

    a_host.x(0) = 1.0;
    a_host.y(0) = 1.0;
    norm_host.x(0) = 1.0;
    norm_host.y(0) = 0.0;
    tan1_host.x(0) = 0.0;
    tan1_host.y(0) = 1.0;
    tan2_host.z(0) = 1.0;

    a_host.x(1) = 1.0;
    a_host.y(1) = 0.0;
    norm_host.x(1) = 0.0;
    norm_host.y(1) = 1.0;
    tan1_host.x(1) = 1.0;
    tan1_host.y(1) = 0.0;
    tan2_host.z(1) = 1.0;

    a_host.x(2) = 1.0;
    a_host.y(2) = 1.0;
    norm_host.x(2) = -1 / Ibis::sqrt(2);
    norm_host.y(2) = 1 / Ibis::sqrt(2);
    tan1_host.x(2) = 1 / Ibis::sqrt(2);
    norm_host.y(2) = 1 / Ibis::sqrt(2);
    tan2_host.z(2) = 1.0;

    a_dev.deep_copy(a_host);
    norm_dev.deep_copy(norm_host);
    tan1_dev.deep_copy(tan1_host);
    tan2_dev.deep_copy(tan2_host);

    transform_to_local_frame(a_dev, norm_dev, tan1_dev, tan2_dev);

    a_host.deep_copy(a_dev);

    CHECK(Kokkos::abs(a_host.x(0) - 1.0) < 1e-14);
    CHECK(Kokkos::abs(a_host.y(0) - 1.0) < 1e-14);

    CHECK(Kokkos::abs(a_host.x(1) - 0.0) < 1e-14);
    CHECK(Kokkos::abs(a_host.y(1) - 1.0) < 1e-14);
}
