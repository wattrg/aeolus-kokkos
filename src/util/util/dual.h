#ifndef DUAL_H
#define DUAL_H

#include <util/real.h>

#include <Kokkos_Core.hpp>

namespace Ibis {

template <typename T>
class Dual {
public:
    // constructors
    KOKKOS_INLINE_FUNCTION
    Dual(T real, T dual) : real_(real), dual_(dual) {}

    KOKKOS_INLINE_FUNCTION
    Dual(T real) : real_(real), dual_(T(0.0)) {}

    KOKKOS_INLINE_FUNCTION
    Dual() : real_(T(0.0)), dual_(T(0.0)) {}

    KOKKOS_INLINE_FUNCTION
    Dual(Dual<T>& other) : real_(other.real_), dual_(other.dual_) {}

    KOKKOS_INLINE_FUNCTION
    Dual(const Dual<T>& other) : real_(other.real_), dual_(other.dual_) {}

    // assignment operators
    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator=(const T& real) {
        this->real_ = real;
        this->dual_ = T(0.0);
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator=(const Dual<T>& other) {
        this->real_ = other.real_;
        this->dual_ = other.dual_;
        return *this;
    }

    // comparison operators
    KOKKOS_INLINE_FUNCTION
    friend bool operator<(const Dual<T>& lhs, const Dual<T>& rhs) {
        return lhs.real_ > rhs.real_;
    }

    KOKKOS_INLINE_FUNCTION
    friend bool operator>(const Dual<T>& lhs, const Dual<T>& rhs) {
        return lhs.real_ > rhs.real_;
    }

    KOKKOS_INLINE_FUNCTION
    friend bool operator<=(const Dual<T>& lhs, const Dual<T>& rhs) {
        return lhs.real_ <= rhs.real_;
    }

    KOKKOS_INLINE_FUNCTION
    friend bool operator>=(const Dual<T>& lhs, const Dual<T>& rhs) {
        return lhs.real_ >= rhs.real_;
    }

    KOKKOS_INLINE_FUNCTION
    friend bool operator==(const Dual<T>& lhs, const Dual<T>& rhs) {
        return lhs.real_ == rhs.real_;
    }

    KOKKOS_INLINE_FUNCTION
    friend bool operator!=(const Dual<T>& lhs, const Dual<T>& rhs) {
        return lhs.real_ != rhs.real_;
    }

    // addition operators
    KOKKOS_INLINE_FUNCTION
    friend Dual<T> operator+(const Dual<T>& lhs, const Dual<T>& rhs) {
        return Dual<T>{lhs.real_ + rhs.real_, lhs.dual_ + rhs.dual_};
    }

    KOKKOS_INLINE_FUNCTION
    friend Dual<T> operator+(const Dual<T>& lhs, const T& rhs) {
        return Dual<T>{lhs.real_ + rhs, lhs.dual_};
    }

    KOKKOS_INLINE_FUNCTION
    friend Dual<T> operator+(const T& lhs, const Dual<T>& rhs) {
        return Dual<T>{lhs + rhs.real_, rhs.dual_};
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator+=(const Dual<T>& other) {
        this->real_ += other.real_;
        this->dual_ += other.dual_;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T> operator+=(const T& re) {
        this->real_ += re;
        return *this;
    }

    // subtraction operators
    KOKKOS_INLINE_FUNCTION
    friend Dual<T> operator-(const Dual<T>& lhs, const Dual<T>& rhs) {
        return Dual<T>{lhs.real_ - rhs.real_, lhs.dual_ - rhs.dual_};
    }

    KOKKOS_INLINE_FUNCTION
    friend Dual<T> operator-(const Dual<T>& lhs, const T& rhs) {
        return Dual<T>{lhs.real_ - rhs, lhs.dual_};
    }

    KOKKOS_INLINE_FUNCTION
    friend Dual<T> operator-(const T& lhs, const Dual<T>& rhs) {
        return Dual<T>{lhs - rhs.real_, rhs.dual_};
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator-=(const Dual<T>& other) {
        this->real_ -= other.real_;
        this->dual_ -= other.dual_;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator-=(T re) {
        this->real_ -= re;
        return *this;
    }

    // negation operator
    KOKKOS_INLINE_FUNCTION
    Dual<T> operator-() const { return Dual<T>{-this->real_, -this->dual_}; }

    // multiplication operators
    KOKKOS_INLINE_FUNCTION
    friend Dual<T> operator*(const Dual<T>& lhs, const Dual<T>& rhs) {
        return Dual<T>{lhs.real_ * rhs.real_,
                       lhs.real_ * rhs.dual_ + lhs.dual_ * rhs.real_};
    }

    KOKKOS_INLINE_FUNCTION
    friend Dual<T> operator*(const Dual<T>& lhs, const T& rhs) {
        return Dual<T>{lhs.real_ * rhs, lhs.dual_ * rhs};
    }

    KOKKOS_INLINE_FUNCTION
    friend Dual<T> operator*(const T& lhs, const Dual<T>& rhs) {
        return Dual<T>{rhs.real_ * lhs, rhs.dual_ * lhs};
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator*=(const Dual<T>& other) {
        this->real_ *= other.real_;
        this->dual_ = this->real_ * other.dual_ + this->dual_ * other.real_;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator*=(T re) {
        this->real_ *= re;
        this->dual_ *= re;
        return *this;
    }

    // division operators
    KOKKOS_INLINE_FUNCTION
    friend Dual<T> operator/(const Dual<T>& num, const Dual<T>& den) {
        return Dual<T>{
            num.real_ / den.real_,
            (num.real_ * den.dual_ - num.dual_ * den.real_) / (den.real_ * den.real_)};
    }

    KOKKOS_INLINE_FUNCTION
    friend Dual<T> operator/(const Dual<T>& num, const T& den) {
        return Dual<T>{num.real_ / den, num.dual_ / den};
    }

    KOKKOS_INLINE_FUNCTION
    friend Dual<T> operator/(const T& num, const Dual<T>& den) {
        return Dual<T>{num / den.real_, -den.dual_ * num / (den.real_ * den.real_)};
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator/=(const Dual<T>& other) {
        this->real_ /= other.real_;
        this->dual_ = this->real_ * other->dual_ - this->dual_ * other.real_;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator/=(T re) {
        this->real_ /= re;
        this->dual_ /= re;
        return *this;
    }

    // some properties of the number
    KOKKOS_INLINE_FUNCTION
    T real() const { return this->real_; }

    KOKKOS_INLINE_FUNCTION
    T dual() const { return this->dual_; }

    KOKKOS_INLINE_FUNCTION
    T& real() { return this->real_; }

    KOKKOS_INLINE_FUNCTION
    T& dual() { return this->dual_; }

    KOKKOS_INLINE_FUNCTION
    T abs() const { return Kokkos::sqrt(this->real_ * this->real_ + dual_ * dual_); }

    KOKKOS_INLINE_FUNCTION
    Dual<T> conjugate() const { return Dual<T>{this->real_, -this->dual_}; }

private:
    T real_;
    T dual_;
};

// stand alone math functions
template <typename T>
KOKKOS_INLINE_FUNCTION Dual<T> sqrt(const Dual<T>& d) {
    T real = Kokkos::sqrt(d.real());
    T dual = d.dual() / (T(2.0) * real);
    return Dual<T>{real, dual};
}

template <typename T>
KOKKOS_INLINE_FUNCTION Dual<T> abs(const Dual<T>& d) {
    return Kokkos::abs(d.real());
    // return Kokkos::sqrt(d.real() * d.real() + d.dual() * d.dual());
}

template <typename T>
KOKKOS_INLINE_FUNCTION Dual<T> pow(const Dual<T>& base, const Dual<T>& power) {
    T real = Kokkos::pow(base.real(), power.real());
    T dual =
        base.dual() * power.real() * Kokkos::pow(base.real(), power.real() - T(1.0)) +
        power.dual() * real * Kokkos::log(base.real());
    return Dual<T>{real, dual};
}

template <typename T>
KOKKOS_INLINE_FUNCTION Dual<T> max(const Dual<T>& d1, const Dual<T>& d2) {
    return d1.real() > d2.real() ? d1 : d2;
}
template <typename T>
KOKKOS_INLINE_FUNCTION Dual<T> max(const T& d1, const Dual<T>& d2) {
    return d1 > d2.real() ? d1 : d2;
}
template <typename T>
KOKKOS_INLINE_FUNCTION Dual<T> max(const Dual<T>& d1, const T& d2) {
    return d1.real() > d2 ? d1 : d2;
}

template <typename T>
KOKKOS_INLINE_FUNCTION Dual<T> min(const Dual<T>& d1, const Dual<T>& d2) {
    return d1.real() < d2.real() ? d1 : d2;
}

template <typename T>
KOKKOS_INLINE_FUNCTION Dual<T> min(const T& d1, const Dual<T>& d2) {
    return d1 < d2.real() ? d1 : d2;
}

template <typename T>
KOKKOS_INLINE_FUNCTION Dual<T> min(const Dual<T>& d1, const T& d2) {
    return d1.real() < d2 ? d1 : d2;
}

template <typename T>
KOKKOS_INLINE_FUNCTION Dual<T> floor(const Dual<T>& d) {
    return Dual<T>{Kokkos::floor(d.real()), Kokkos::floor(d.dual())};
}

template <typename T>
KOKKOS_INLINE_FUNCTION Dual<T> ceil(const Dual<T>& d) {
    return Dual<T>{Kokkos::ceil(d.real()), Kokkos::ceil(d.dual())};
}

template <typename T>
KOKKOS_INLINE_FUNCTION Dual<T> copysign(const Dual<T>& mag, const Dual<T>& sign) {
    T real = Kokkos::copysign(mag.real(), sign.real());
    T dual = real == mag.real() ? mag.dual() : -mag.dual();
    return Dual<T>{real, dual};
}

template <typename T>
KOKKOS_INLINE_FUNCTION bool isnan(const Dual<T>& d) {
    return Kokkos::isnan(d.real()) || Kokkos::isnan(d.dual());
}

template <typename T>
KOKKOS_INLINE_FUNCTION bool isinf(const Dual<T>& d) {
    return Kokkos::isinf(d.real()) || Kokkos::isinf(d.dual());
}

template <typename T>
KOKKOS_INLINE_FUNCTION T real_part(const Dual<T>& d) {
    return d.real();
}

template <typename T>
KOKKOS_INLINE_FUNCTION T& real_part(Dual<T>& d) {
    return d.real();
}

template <typename T>
KOKKOS_INLINE_FUNCTION T dual_part(const Dual<T>& d) {
    return d.dual();
}

template <typename T>
KOKKOS_INLINE_FUNCTION T& dual_part(Dual<T>& d) {
    return d.dual();
}

typedef Dual<real> dual;
}  // namespace Ibis

#endif
