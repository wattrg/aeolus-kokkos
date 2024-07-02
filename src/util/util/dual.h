#ifndef DUAL_H
#define DUAL_H

#include <Kokkos_Core.hpp>

namespace Ibis {

template <typename T>
class Dual<T> {
public:
    // constructors
    KOKKOS_INLINE_FUNCTION
    Dual(T re, T dual) : re_(re), dual_(dual) {}

    KOKKOS_INLINE_FUNCTION
    Dual(T re) : re_(re), dual_(0.0) {}

    KOKKOS_INLINE_FUNCTION
    Dual(Dual<T>& other) : re_(other.re_), dual_(other.dual_) {}

    // assignment operator
    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator=(T re) {
        this->re_ = re;
        this->dual_ = 0.0;
        return *this;
    }

    // addition operators
    KOKKOS_INLINE_FUNCTION
    Dual<T> operator+(Dual<T>& other) {
        return Dual<T>{this->re_ + other.re_, this->dual_ + other.dual_};
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator+=(Dual<T>& other) {
        this->re_ += other.re_;
        this->dual_ += other.dual_;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T> operator+(T re) { return Dual<T>{this->re_ + re, this->dual_}; }

    KOKKOS_INLINE_FUNCTION
    Dual<T> operator+=(T re) {
        this->_re_ += re;
        return *this;
    }

    // subtraction operators
    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator-(Dual<T>& other) {
        return Dual<T>{this->re_ - other.re_, this->dual_ - other.dual_};
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator-=(Dual<T>& other) {
        this->re_ -= other.re_;
        this->dual_ -= other.dual_;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T> operator-(T re) { return Dual<T>{this->re_ - re, this->dual_}; }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator-=(T re) {
        this->re_ -= re;
        return *this;
    }

    // multiplication operators
    KOKKOS_INLINE_FUNCTION
    Dual<T> operator*(Dual<T>& other) {
        return Dual<T>{this->re_ * other.re_,
                       this->re_ * other.dual_ + this->dual_ * other.re_};
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator*=(Dual<T>& other) {
        this->re_ *= other.re_;
        this->dual_ = this->re_ * other.dual_ + this->dual_ * other.re_;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator*=(T re) {
        this->re_ *= re;
        this->dual_ *= re;
        return *this;
    }

    // division operators
    KOKKOS_INLINE_FUNCTION
    Dual<T> operator/(Dual<T>& other) {
        return Dual<T>{this->re_ / other.re_,
                       this->re_ * other->dual_ - this->dual_ * other.re_};
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator/=(Dual<T>& other) {
        this->re_ /= other.re_;
        this->dual_ = this->re_ * other->dual_ - this->dual_ * other.re_;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Dual<T> operator/(T re) { return Dual<T>{this->re_ / re, this->dual_ / re}; }

    KOKKOS_INLINE_FUNCTION
    Dual<T>& operator/=(T re) {
        this->re_ /= re;
        this->dual_ /= re;
        return *this;
    }

private:
    T re_;
    T dual_;
};

typedef Dual<double> dual;
}  // namespace Ibis

#endif
