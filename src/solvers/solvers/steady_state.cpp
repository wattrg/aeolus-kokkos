#include <finite_volume/conserved_quantities.h>
#include <finite_volume/finite_volume.h>
#include <finite_volume/primative_conserved_conversion.h>
#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <simulation/simulation.h>
#include <solvers/cfl.h>
#include <solvers/steady_state.h>
#include <solvers/transient_linear_system.h>

SteadyStateLinearisation::SteadyStateLinearisation(std::shared_ptr<Sim<Ibis::dual>>& sim,
                                                   ConservedQuantities<Ibis::dual> cq,
                                                   FlowStates<Ibis::dual> fs) {
    sim_ = sim;
    cq_ = cq;
    fs_ = fs;

    n_cells_ = sim_->grid.num_total_cells();
    n_cons_ = cq_.n_conserved();
    n_vars_ = n_cells_ * n_cons_;
    size_t dim = sim_->grid.dim();

    rhs_ = Ibis::Vector<Ibis::real>{"SteadyStateLinearisation::rhs", n_vars_};
    fs_tmp_ = FlowStates<Ibis::dual>{n_cells_};
    cq_tmp_ = ConservedQuantities<Ibis::dual>{n_cells_, dim};
}

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
            const size_t vector_idx = cell_i * n_cons;
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
            const size_t vector_idx = cell_i * n_cons;
            for (size_t cons_i = 0; cons_i < n_cons; cons_i++) {
                rhs(vector_idx + cons_i) = Ibis::real_part(residuals(cell_i, cons_i));
            }
        });
}

SteadyState::SteadyState(json config, GridBlock<Ibis::dual> grid, std::string grid_dir,
                         std::string flow_dir)
    : Solver(grid_dir, flow_dir) {
    sim_ = std::shared_ptr<Sim<Ibis::dual>>{new Sim<Ibis::dual>(grid, config)};

    size_t n_cells = sim_->grid.num_total_cells();
    size_t dim = sim_->grid.dim();
    fs_ = FlowStates<Ibis::dual>(n_cells);
    cq_ = ConservedQuantities<Ibis::dual>(n_cells, dim);

    // set up the linear system and non-linear solver
    std::unique_ptr<PseudoTransientLinearSystem> system =
        std::unique_ptr<PseudoTransientLinearSystem>(
            new SteadyStateLinearisation(sim_, cq_, fs_));
    auto cfl = make_cfl_schedule(config.at("cfl"));
    jfnk_ = Jfnk(std::move(system), std::move(cfl), config);

    // configuration
    print_frequency_ = config.at("print_frequency");
    plot_frequency_ = config.at("plot_frequency");
    diagnostics_frequency_ = config.at("diagnostics_frequency");

    // I/O
    FlowFormat flow_format = string_to_flow_format((config.at("io").at("flow_format")));
    // io_ = FVIO<Ibis::real>(flow_format, flow_format, 1);
}

int SteadyState::initialise() {
    jfnk_.initialise();
    return 0;
}

int SteadyState::take_step() {
    jfnk_.step(sim_, cq_, fs_);
    return 0;
}

bool SteadyState::print_this_step(unsigned int step) {
    return (step != 0 && step % print_frequency_ == 0);
}

bool SteadyState::residuals_this_step(unsigned int step) {
    return ((diagnostics_frequency_ > 0) && (step != 0) &&
            (step % diagnostics_frequency_ == 0));
}

bool SteadyState::plot_this_step(unsigned int step) {
    return (step != 0 && plot_frequency_ % plot_frequency_ == 0);
}

int SteadyState::plot_solution(unsigned int step) {
    Ibis::real t = Ibis::real(step);
    int result =
        io_.write(fs_, sim_->fv, sim_->grid, sim_->gas_model, sim_->trans_prop, t);
    spdlog::info("  written flow solution: step {}", step);
    return result;
}

void SteadyState::print_progress(unsigned int step, Ibis::real wc) {
    spdlog::info("  step: {:>8}, global residual {:.6e}, wc = {:.1f}s", step, -1.0, wc);
}

bool SteadyState::stop_now(unsigned int step) { return (step >= max_step() - 1); }

std::string SteadyState::stop_reason(unsigned int step) {
    if (step > max_step()) return "reached max_step";
    if (jfnk_.global_residual() < jfnk_.target_residual()) {
        return "reached target residual";
    }
    return "Shouldn't reach here";
}
