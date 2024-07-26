#ifndef LA_NORM_H
#define LA_HORM_H

#include <util/numeric_types.h>
#include <util/types.h>

#include <Kokkos_Core.hpp>

namespace Ibis {

template <typename T>
T norm2(const Ibis::Vector<T>& vec);

template <typename T>
T norm2_squared(const Ibis::Vector<T>& vec);

}  // namespace Ibis

#endif
