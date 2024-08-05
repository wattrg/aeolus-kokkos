#include <finite_volume/conserved_quantities.h>
#include <finite_volume/primative_conserved_conversion.h>
#include <solvers/jfnk.h>

Jfnk::Jfnk(std::shared_ptr<PseudoTransientLinearSystem> system,
           std::unique_ptr<CflSchedule>&& cfl, json config) {
    max_steps_ = config.at("max_steps");
    tolerance_ = config.at("tolerance");
    gmres_ = Gmres(system, config.at("linear_solver"));
    cfl_ = std::move(cfl);
    system_ = system;
    dU_ = Ibis::Vector<Ibis::real>{"dU", system_->num_vars()};
}

int Jfnk::initialise() { return 0; }

void Jfnk::step(std::shared_ptr<Sim<Ibis::dual>>& sim,
                ConservedQuantities<Ibis::dual>& cq, FlowStates<Ibis::dual>& fs) {
    // dU is the change in the solution for the step,
    // our initial guess for it is zero
    dU_.zero();
    Ibis::real cfl = cfl_->eval(0.0);
    stable_dt_ = sim->fv.estimate_dt(fs, sim->grid, sim->gas_model, sim->trans_prop);
    system_->set_pseudo_time_step(cfl * stable_dt_);
    gmres_.solve(system_, dU_);
    apply_update_(sim, cq, fs);
}

void Jfnk::apply_update_(std::shared_ptr<Sim<Ibis::dual>>& sim,
                         ConservedQuantities<Ibis::dual>& cq,
                         FlowStates<Ibis::dual>& fs) {
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
    conserved_to_primatives(cq, fs, sim->gas_model);
}
