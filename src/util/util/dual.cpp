#include <doctest/doctest.h>
#include <util/dual.h>

TEST_CASE("Dual::addition") {
    Ibis::dual x{3.0, 1.0};
    Ibis::dual y{2.0, 3.0};
    auto d = x + y;

    CHECK(d.real() == doctest::Approx(5.0));
    CHECK(d.dual() == doctest::Approx(4.0));
}

TEST_CASE("Dual::+=") {
    Ibis::dual x{3.0, 1.0};
    Ibis::dual y{2.0, 3.0};
    x += y;

    CHECK(x.real() == doctest::Approx(5.0));
    CHECK(x.dual() == doctest::Approx(4.0));
}

TEST_CASE("Dual::addition_real") {
    Ibis::dual x{3.0, 1.0};
    Ibis::real y = 2.0;
    auto d = x + y;

    CHECK(d.real() == doctest::Approx(5.0));
    CHECK(d.dual() == doctest::Approx(1.0));
}

TEST_CASE("Dual::subtracton") {
    Ibis::dual x{3.0, 1.0};
    Ibis::dual y{2.0, 3.0};
    auto d = x - y;

    CHECK(d.real() == doctest::Approx(1.0));
    CHECK(d.dual() == doctest::Approx(-2.0));
}

TEST_CASE("Dual::-=") {
    Ibis::dual x{3.0, 1.0};
    Ibis::dual y{2.0, 3.0};
    x -= y;

    CHECK(x.real() == doctest::Approx(1.0));
    CHECK(x.dual() == doctest::Approx(-2.0));
}

TEST_CASE("Dual::subtract_real") {
    Ibis::dual x{3.0, 1.0};
    Ibis::real y = 2.0;
    auto d = x - y;

    CHECK(d.real() == doctest::Approx(1.0));
    CHECK(d.dual() == doctest::Approx(1.0));
}

TEST_CASE("Dual::real_subtract_dual") {
    Ibis::dual x{3.0, 1.0};
    Ibis::real y = 2.0;
    auto d = y - x;

    CHECK(d.real() == doctest::Approx(-1.0));
    CHECK(d.dual() == doctest::Approx(-1.0));
}

TEST_CASE("Dual::multiplicaton") {
    Ibis::dual x{3.0, 1.0};
    Ibis::dual y{2.0, 3.0};
    auto d = x * y;

    CHECK(d.real() == doctest::Approx(6.0));
    CHECK(d.dual() == doctest::Approx(11.0));
}

TEST_CASE("Dual::*=") {
    Ibis::dual x{3.0, 1.0};
    Ibis::dual y{2.0, 3.0};
    x *= y;
    
    CHECK(x.real() == doctest::Approx(6.0));
    CHECK(x.dual() == doctest::Approx(11.0));
}

TEST_CASE("Dual::dual_multiply_real") {
    Ibis::dual x{3.0, 1.0};
    Ibis::real y = 2.0;
    auto d = x * y;

    CHECK(d.real() == doctest::Approx(6.0));
    CHECK(d.dual() == doctest::Approx(2.0));
}

TEST_CASE("Dual::dual_multiply_eq_real") {
    Ibis::dual x{3.0, 1.0};
    Ibis::real y = 2.0;
    x *= y;

    CHECK(x.real() == doctest::Approx(6.0));
    CHECK(x.dual() == doctest::Approx(2.0));
}

TEST_CASE("Dual::real_multiply_dual") {
    Ibis::dual x{3.0, 1.0};
    Ibis::real y = 2.0;
    auto d = y * x;

    CHECK(d.real() == doctest::Approx(6.0));
    CHECK(d.dual() == doctest::Approx(2.0));
}

TEST_CASE("Dual::division") {
    Ibis::dual x{3.0, 1.0};
    Ibis::dual y{2.0, 3.0};
    auto d = x / y;

    CHECK(d.real() == doctest::Approx(1.5));
    CHECK(d.dual() == doctest::Approx(-1.75));
}

TEST_CASE("Dual::/=") {
    Ibis::dual x{3.0, 1.0};
    Ibis::dual y{2.0, 3.0};
    x /= y;

    CHECK(x.real() == doctest::Approx(1.5));
    CHECK(x.dual() == doctest::Approx(-1.75));
}

TEST_CASE("Dual::dual_divde_real") {
    Ibis::dual x{3.0, 1.0};
    Ibis::real y = 2.0;
    auto d = x / y;

    CHECK(d.real() == doctest::Approx(1.5));
    CHECK(d.dual() == doctest::Approx(0.5));
}

TEST_CASE("Dual::dual_divde_eq_real") {
    Ibis::dual x{3.0, 1.0};
    Ibis::real y = 2.0;
    x /= y;

    CHECK(x.real() == doctest::Approx(1.5));
    CHECK(x.dual() == doctest::Approx(0.5));
}

TEST_CASE("Dual::real_divde_dual") {
    Ibis::dual x{4.0, 1.0};
    Ibis::real y = 2.0;
    auto d = y / x;

    CHECK(d.real() == doctest::Approx(0.5));
    CHECK(d.dual() == doctest::Approx(-0.125));
}

TEST_CASE("Dual::sqrt") {
    Ibis::dual x{9.0, 3.0};
    auto d = Ibis::sqrt(x);

    CHECK(d.real() == doctest::Approx(3.0));
    CHECK(d.dual() == doctest::Approx(0.5));
}

TEST_CASE("Dual::greater") {
    Ibis::dual x{3.0, 1.0};
    Ibis::dual y{2.0, 3.0};
    CHECK_GT(x, y);
}

TEST_CASE("Dual::lesser") {
    Ibis::dual x{3.0, 1.0};
    Ibis::dual y{2.0, 3.0};
    CHECK_LT(y, x);
}

TEST_CASE("Dual::x^2+2x") {
    Ibis::dual x{2.0, 1.0};
    auto y = x * x + 2 * x;
    CHECK(y.real() == doctest::Approx(8.0));
    CHECK(y.dual() == doctest::Approx(6.0));
}

TEST_CASE("Dual::x^2-sqrt(x)") {
    Ibis::dual x{9.0, 1.0};
    auto y = x * x - Ibis::sqrt(x);
    CHECK(y.real() == doctest::Approx(78.0));
    CHECK(y.dual() == doctest::Approx(18.0 - 1.0 / 6.0));
}

TEST_CASE("Dual::diff_abs") {
    Ibis::dual x{-1.0, 1.0};
    Ibis::dual y = Ibis::abs(x);

    CHECK(y.real() == doctest::Approx(1.0));
    CHECK(y.dual() == doctest::Approx(-1.0));
}

TEST_CASE("Dual::diff_abs_2") {
    Ibis::dual x{1.0, 1.0};
    Ibis::dual y = Ibis::abs(x);

    CHECK(y.real() == doctest::Approx(1.0));
    CHECK(y.dual() == doctest::Approx(1.0));
}

TEST_CASE("Dual::abs(x^2)") {
    Ibis::dual x{-1.0, 1.0};
    Ibis::dual y = Ibis::abs(x * x);

    CHECK(y.real() == doctest::Approx(1.0));
    CHECK(y.dual() == doctest::Approx(-2.0));
}

TEST_CASE("Dual::abs(x^2)_2") {
    Ibis::dual x{1.0, 1.0};
    Ibis::dual y = Ibis::abs(x * x);

    CHECK(y.real() == doctest::Approx(1.0));
    CHECK(y.dual() == doctest::Approx(2.0));
}
