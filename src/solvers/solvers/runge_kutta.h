#ifndef RUNGE_KUTTA_H
#define RUNGE_KUTTA_H

#include <finite_volume/conserved_quantities.h>
#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <grid/grid.h>
#include <io/io.h>
#include <solvers/cfl.h>
#include <solvers/solver.h>
#include <util/numeric_types.h>

#include <memory>

class ButcherTableau {
public:
    ButcherTableau() {}

    ButcherTableau(std::vector<std::vector<Ibis::real>> a, std::vector<Ibis::real> b,
                   std::vector<Ibis::real> c)
        : a_(a), b_(b), c_(c) {
        num_stages_ = b_.size();
    }

    ButcherTableau(json config)
        : ButcherTableau(config.at("a"), config.at("b"), config.at("c")) {}

    Ibis::real a(size_t i, size_t j);
    Ibis::real b(size_t i);
    Ibis::real c(size_t i);
    size_t num_stages();

private:
    std::vector<std::vector<Ibis::real>> a_;
    std::vector<Ibis::real> b_;
    std::vector<Ibis::real> c_;
    size_t num_stages_;
};

class RungeKutta : public Solver {
public:
    RungeKutta(json config, GridBlock<Ibis::real>& grid, std::string grid_dir,
               std::string flow_dir);

    ~RungeKutta() {}

    int solve();

private:
    // configuration
    Ibis::real max_time_;
    unsigned int max_step_;
    unsigned int print_frequency_;
    Ibis::real plot_frequency_;
    Ibis::real residual_frequency_;
    int residuals_every_n_steps_;
    int plot_every_n_steps_;
    std::unique_ptr<CflSchedule> cfl_;
    Ibis::real dt_init_;

private:
    // progress
    Ibis::real time_since_last_plot_;
    Ibis::real time_since_last_residual_;
    Ibis::real t_;
    Ibis::real dt_;
    Ibis::real stable_dt_;

private:
    // input/output
    FVIO<Ibis::real> io_;

private:
    // implementation
    int initialise();
    int finalise();
    int take_step();
    void estimate_dt();
    bool print_this_step(unsigned int step);
    bool plot_this_step(unsigned int step);
    int plot_solution(unsigned int step);
    void print_progress(unsigned int step, Ibis::real wc);
    std::string stop_reason(unsigned int step);
    bool stop_now(unsigned int step);
    int max_step() const { return max_step_; }
    int count_bad_cells() { return fv_.count_bad_cells(flow_, grid_.num_cells()); }

    // this computes the L2 norms of the time derivates evaluated
    // at the beginning of the previous step (essential whatever is in k_[0]).
    // It should be called after taking a step, so the
    ConservedQuantitiesNorm<Ibis::real> L2_norms();

    bool residuals_this_step(unsigned int step);
    bool write_residuals(unsigned int step);

private:
    // memory
    FlowStates<Ibis::real> flow_;
    ConservedQuantities<Ibis::real> conserved_quantities_;
    std::vector<ConservedQuantities<Ibis::real>> k_;
    ConservedQuantities<Ibis::real> k_tmp_;
    FlowStates<Ibis::real> flow_tmp_;

    // butcher tableau
    ButcherTableau tableau_;

private:
    // spatial discretisation
    GridBlock<Ibis::real> grid_;
    FiniteVolume<Ibis::real> fv_;

private:
    IdealGas<Ibis::real> gas_model_;
    TransportProperties<Ibis::real> trans_prop_;
};

#endif
