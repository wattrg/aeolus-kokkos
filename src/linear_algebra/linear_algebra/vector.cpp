#include <linear_algebra/vector.h>

namespace Ibis {

template <typename T>
T norm2_squared(const Vector<T>& vec) {
    T norm2;
    Kokkos::parallel_reduce(
        "Vector::norm2", vec.extent(0),
        KOKKOS_LAMBDA(const size_t i, T& utd) {
            T value = vec(i);
            utd += value * value;
        },
        Kokkos::Sum<T>(norm2));
    return norm2;
}
template real norm2_squared(const Vector<real>&);

template <typename T>
T norm2(const Vector<T>& vec) {
    return Ibis::sqrt(norm2_squared(vec));
}
template real norm2(const Vector<real>&);

template <typename T>
void scale_in_place(Vector<T>& vec, const T factor) {
    Kokkos::parallel_for(
        "Ibis::Vector::scale_in_place", vec.extent(0),
        KOKKOS_LAMBDA(const size_t i) { vec(i) *= factor; });
}
template void scale_in_place(Vector<real>&, const real);


template <typename T>
T dot(const Vector<T>& vec1, const Vector<T>& vec2) {
    assert(vec1.extent(0) == vec2.extent(0));
    T dot_product;
    Kokkos::parallel_reduce(
        "Ibis::Vector::dot", vec1.extent(0),
        KOKKOS_LAMBDA(const size_t i, T& utd) { utd += vec1(i) * vec2(i); },
        Kokkos::Sum<T>(dot_product));
    return dot_product;
}
template real dot(const Vector<real>&, const Vector<real>&);

}  // namespace Ibis
