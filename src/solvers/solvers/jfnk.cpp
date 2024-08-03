#include <solvers/jfnk.h>

#include "finite_volume/conserved_quantities.h"

Jfnk::Jfnk(std::unique_ptr<LinearSystem>&& system, std::unique_ptr<CflSchedule>&& cfl,
           json config) {
    max_steps_ = config.at("max_steps");
    gmres_ = Gmres(system, config.at("gmres"));
    cfl_ = std::move(cfl);
    system_ = std::move(system);
}

void Jfnk::step(ConservedQuantities<Ibis::dual>& cq) {
    // dU is the change in the solution for the step,
    // our initial guess for it is zero
    dU_.zero();
    gmres_.solve(system_, dU_);
    apply_update_(cq);
}

void Jfnk::apply_update_(ConservedQuantities<Ibis::dual>& cq) {
    auto dU = dU_;
    size_t n_cells = cq.size();
    size_t n_cons = cq.n_conserved();
    Kokkos::parallel_for(
        "Jfnk::apply_update", n_cells, KOKKOS_LAMBDA(const size_t cell_i) {
            const size_t vector_idx = cell_i * n_cons;
            for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                cq(cell_i, cons_i) += dU(vector_idx + cons_i);
            }
        });
}
