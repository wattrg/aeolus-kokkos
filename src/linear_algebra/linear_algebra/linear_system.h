#ifndef LINEAR_SYSTEM_H
#define LINEAR_SYSTEM_H

#include <finite_volume/conserved_quantities.h>
#include <simulation/simulation.h>
#include <util/numeric_types.h>
#include <linear_algebra/vector.h>

class LinearSystem {
public:
    LinearSystem(){};

    virtual ~LinearSystem() {}

    virtual void matrix_vector_product(Ibis::Vector<Ibis::real>& vec,
                                       Ibis::Vector<Ibis::real>& result) = 0;

    virtual void eval_rhs() = 0;

    KOKKOS_INLINE_FUNCTION
    virtual Ibis::real& rhs(const size_t i) const = 0;

    KOKKOS_INLINE_FUNCTION
    virtual Ibis::real& rhs(const size_t cell_i, const size_t cons_i) const = 0;

    virtual Ibis::Vector<Ibis::real>& rhs() = 0;

    virtual size_t num_vars() const = 0;
};

#endif
