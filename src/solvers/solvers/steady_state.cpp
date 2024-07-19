#include <finite_volume/conserved_quantities.h>
#include <finite_volume/finite_volume.h>
#include <finite_volume/primative_conserved_conversion.h>
#include <gas/transport_properties.h>
#include <solvers/steady_state.h>

SteadyStateLinearisation::SteadyStateLinearisation(const size_t n_cells,
                                                   const size_t n_cons,
                                                   const size_t dim) {
    n_cells_ = n_cells;
    n_cons_ = n_cons;
    dim_ = dim;
    n_vars_ = n_cells_ * n_cons_;
    residuals_ = ConservedQuantities<Ibis::dual>(n_cells_, dim_);
}

void SteadyStateLinearisation::matrix_vector_product(
    FiniteVolume<Ibis::dual>& fv, FlowStates<Ibis::dual>& fs,
    ConservedQuantities<Ibis::dual>& cq, const GridBlock<Ibis::dual>& grid,
    IdealGas<Ibis::dual>& gas_model, TransportProperties<Ibis::dual>& trans_prop,
    Field<Ibis::real>& vec) {
    // set the dual components of the conserved quantities/flow states
    size_t n_cons = n_cons_;
    auto residuals = residuals_;
    Ibis::real dt_star = dt_star_;
    Kokkos::parallel_for(
        "SteadyStateLinearisation::set_dual", n_cells_, KOKKOS_LAMBDA(const int cell_i) {
            const int vector_idx = cell_i * n_cons;
            for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                cq(cell_i, cons_i) = vec(vector_idx + cons_i);
            }
        });

    // convert the conserved quantities to primatives, so that the primatives
    // have dual components set.
    conserved_to_primatives(cq, fs, gas_model);

    // evaluate the residuals
    fv.compute_dudt(fs, grid, residuals, gas_model, trans_prop);

    // set the components of vec to the dual component of dudt
    Kokkos::parallel_for(
        "SteadyStateLinearisation::set_vector", n_cells_,
        KOKKOS_LAMBDA(const int cell_i) {
            const int vector_idx = cell_i * n_cons;
            for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                vec(vector_idx + cons_i) =
                    1 / dt_star - Ibis::dual_part(residuals(cell_i, cons_i));
            }
        });
}

void SteadyStateLinearisation::eval_rhs(FiniteVolume<Ibis::dual>& fv,
                                        FlowStates<Ibis::dual>& fs,
                                        const GridBlock<Ibis::dual>& grid,
                                        IdealGas<Ibis::dual>& gas_model,
                                        TransportProperties<Ibis::dual>& trans_prop,
                                        Field<Ibis::real>& vec) {
    fv.compute_dudt(fs, grid, residuals_, gas_model, trans_prop);

    ConservedQuantities<Ibis::dual> residuals = residuals_;
    size_t n_cons = n_cons_;
    Kokkos::parallel_for(
        "SteadyStateLinearisation::eval_rhs", n_cells_, KOKKOS_LAMBDA(const int cell_i) {
            const int vector_idx = cell_i * n_cons;
            for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                vec(vector_idx + cons_i) = Ibis::real_part(residuals(cell_i, cons_i));
            }
        });
}
