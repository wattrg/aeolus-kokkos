#ifndef STEADY_STATE_SOLVER_H
#define STEADY_STATE_SOLVER_H

#include <linear_algebra/linear_system.h>

#include "finite_volume/conserved_quantities.h"
#include "gas/flow_state.h"

class SteadyStateLinearisation : public SystemLinearisation {
public:
    // Construction / destruction
    SteadyStateLinearisation(const size_t n_cells, const size_t n_cons, const size_t dim);
    ~SteadyStateLinearisation() {}

    // SystemLinearisation interface
    void matrix_vector_product(FiniteVolume<Ibis::dual>& fv, FlowStates<Ibis::dual>& fs,
                               ConservedQuantities<Ibis::dual>& cq,
                               const GridBlock<Ibis::dual>& grid,
                               IdealGas<Ibis::dual>& gas_model,
                               TransportProperties<Ibis::dual>& trans_prop,
                               Field<Ibis::real>& vec);

    void eval_rhs(FiniteVolume<Ibis::dual>& fv, FlowStates<Ibis::dual>& fs,
                  const GridBlock<Ibis::dual>& grid, IdealGas<Ibis::dual>& gas_model,
                  TransportProperties<Ibis::dual>& trans_prop,
                  Field<Ibis::real>& vec) = 0;

public:
    // some specific methods
    void set_pseudo_time_step(Ibis::real dt_star);

private:
    Ibis::real dt_star_;

    // memory
    size_t n_cells_;
    size_t n_cons_;
    size_t n_vars_;
    size_t dim_;
    ConservedQuantities<Ibis::dual> residuals_;
};

class SteadyState {
    //
};

#endif
