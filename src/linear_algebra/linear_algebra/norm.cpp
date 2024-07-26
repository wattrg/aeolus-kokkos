#include <linear_algebra/norm.h>

namespace Ibis {

template <typename T>
T norm2_squared(const Ibis::Vector<T>& vec) {
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
template Ibis::real norm2_squared(const Ibis::Vector<Ibis::real>& vec);

template <typename T>
T norm2(const Ibis::Vector<T>& vec) {
    return Ibis::sqrt(norm2_squared(vec));
}
template Ibis::real norm2(const Ibis::Vector<Ibis::real>& vec);

}  // namespace Ibis
