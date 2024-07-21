#ifndef STEADY_STATE_SOLVER_H
#define STEADY_STATE_SOLVER_H

#include <finite_volume/conserved_quantities.h>
#include <gas/flow_state.h>
#include <linear_algebra/linear_system.h>
#include <solvers/solver.h>

#include "gas/transport_properties.h"
#include "io/io.h"
#include "solvers/cfl.h"
#include "util/dual.h"

class SteadyStateLinearisation : public SystemLinearisation {
public:
    // Construction / destruction
    SteadyStateLinearisation(const size_t n_cells, const size_t n_cons, const size_t dim);
    ~SteadyStateLinearisation() {}

    // SystemLinearisation interface
    void matrix_vector_product(FiniteVolume<Ibis::dual>& fv,
                               ConservedQuantities<Ibis::dual>& cq,
                               const GridBlock<Ibis::dual>& grid,
                               IdealGas<Ibis::dual>& gas_model,
                               TransportProperties<Ibis::dual>& trans_prop,
                               Ibis::Vector<Ibis::real>& vec);

    void eval_rhs(FiniteVolume<Ibis::dual>& fv, FlowStates<Ibis::dual>& fs,
                  const GridBlock<Ibis::dual>& grid, IdealGas<Ibis::dual>& gas_model,
                  TransportProperties<Ibis::dual>& trans_prop,
                  ConservedQuantities<Ibis::dual>& residuals, Ibis::Vector<Ibis::real>& vec);

    size_t num_vars() const { return n_vars_; };

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
    FlowStates<Ibis::dual> fs_tmp_;
};

class SteadyState : public Solver {
public:
    SteadyState(json config, GridBlock<Ibis::dual>& grid, std::string grid_dir,
                std::string flow_dir);

    ~SteadyState() {}

    int solve();

private:
    // configuration
    unsigned int max_nonlinear_steps_;
    unsigned int print_frequency_;
    unsigned int plot_frequency_;
    unsigned int diagnostics_frequency_;
    std::unique_ptr<CflSchedule> cfl_;

    // progress
    Ibis::real stable_dt_;
    Ibis::real dt_star_;

    // input/output
    FVIO<Ibis::dual> io_;

    // implementation
    int initialise();
    int finalise();
    int take_step();
    bool print_this_step(unsigned int step);
    bool plot_this_step(unsigned int step);
    int plot_solution(unsigned int step);
    void print_progress(unsigned int step, Ibis::real wc);
    std::string stop_reason(unsigned int step);
    bool stop_now(unsigned int step);
    int max_step() const { return max_nonlinear_steps_; }
    int count_bad_cells() { return fv_.count_bad_cells(fs_, grid_.num_cells()); }

private:
    // memory
    ConservedQuantities<Ibis::dual> cq_;
    ConservedQuantities<Ibis::dual> residuals_;
    FlowStates<Ibis::dual> fs_;

    // The RHS vector
    Ibis::Vector<Ibis::dual> rhs_;

    // spatial discretisation
    GridBlock<Ibis::dual> grid_;
    FiniteVolume<Ibis::dual> fv_;

    // gas models
    IdealGas<Ibis::dual> gas_model_;
    TransportProperties<Ibis::dual> trans_prop_;

    // linearisation
    SteadyStateLinearisation linearisation_;
};

#endif
