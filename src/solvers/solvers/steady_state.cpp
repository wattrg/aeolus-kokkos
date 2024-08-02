#include <finite_volume/conserved_quantities.h>
#include <finite_volume/finite_volume.h>
#include <finite_volume/primative_conserved_conversion.h>
#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <simulation/simulation.h>
#include <solvers/steady_state.h>

// SteadyStateLinearisation::SteadyStateLinearisation(const size_t n_cells,
//                                                    const size_t n_cons,
//                                                    const size_t dim) {
//     n_cells_ = n_cells;
//     n_cons_ = n_cons;
//     dim_ = dim;
//     n_vars_ = n_cells_ * n_cons_;
//     residuals_ = ConservedQuantities<Ibis::dual>(n_cells_, dim_);
//     fs_tmp_ = FlowStates<Ibis::dual>(n_cells_);
// }

void SteadyStateLinearisation::matrix_vector_product(Ibis::Vector<Ibis::real>& vec,
                                                     Ibis::Vector<Ibis::real>& result) {
    // set the dual components of the conserved quantities
    size_t n_cons = n_cons_;
    auto residuals = residuals_;
    Ibis::real dt_star = dt_star_;
    auto cq_tmp = cq_tmp_;
    auto cq = cq_;
    Kokkos::parallel_for(
        "SteadyStateLinearisation::set_dual", n_cells_, KOKKOS_LAMBDA(const int cell_i) {
            const int vector_idx = cell_i * n_cons;
            for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                cq_tmp(cell_i, cons_i).real() = cq(cell_i, cons_i).real();
                cq_tmp(cell_i, cons_i).dual() = vec(vector_idx + cons_i);
            }
        });

    // convert the conserved quantities to primatives, ready to evaluate the residuals
    conserved_to_primatives(cq_tmp_, fs_tmp_, sim_->gas_model);

    // evaluate the residuals
    sim_->fv.compute_dudt(fs_tmp_, sim_->grid, residuals, sim_->gas_model,
                          sim_->trans_prop);

    // set the components of vec to the dual component of dudt
    Kokkos::parallel_for(
        "SteadyStateLinearisation::set_vector", n_cells_,
        KOKKOS_LAMBDA(const int cell_i) {
            const int vector_idx = cell_i * n_cons;
            for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                result(vector_idx + cons_i) =
                    1 / dt_star - Ibis::dual_part(residuals(cell_i, cons_i));
            }
        });
}

void SteadyStateLinearisation::eval_rhs() {
    sim_->fv.compute_dudt(fs_, sim_->grid, residuals_, sim_->gas_model, sim_->trans_prop);

    size_t n_cons = n_cons_;
    auto rhs = rhs_;
    auto residuals = residuals_;
    Kokkos::parallel_for(
        "SteadyStateLinearisation::eval_rhs", n_cells_, KOKKOS_LAMBDA(const int cell_i) {
            const int vector_idx = cell_i * n_cons;
            for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                rhs(vector_idx + cons_i) = Ibis::real_part(residuals(cell_i, cons_i));
            }
        });
}

SteadyState::SteadyState(json config, GridBlock<Ibis::dual>& grid, std::string grid_dir, std::string flow_dir) : Solver(grid_dir, flow_dir) {
    sim_ = std::shared_ptr<Sim<Ibis::dual>> {Sim<Ibis::dual>(std::move(grid), config)};

    size_t number_cells = sim_->grid.num_total_cells();
    size_t dim = sim_->grid.dim();
    fs_ = FlowStates<Ibis::dual>(number_cells);
    cq_ = ConservedQuantities<Ibis::dual>(number_cells, dim);

    SteadyStateLinearisation system = SteadyStateLinearisation(sim_, cq_, fs_);
}
