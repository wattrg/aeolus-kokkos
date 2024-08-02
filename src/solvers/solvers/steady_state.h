#ifndef STEADY_STATE_SOLVER_H
#define STEADY_STATE_SOLVER_H

#include <finite_volume/conserved_quantities.h>
#include <gas/flow_state.h>
#include <io/io.h>
#include <linear_algebra/linear_system.h>
#include <simulation/simulation.h>
#include <solvers/cfl.h>
#include <solvers/jfnk.h>
#include <solvers/solver.h>

class SteadyStateLinearisation : public LinearSystem {
public:
    // Construction / destruction
    SteadyStateLinearisation(std::shared_ptr<Sim<Ibis::dual>>& sim, ConservedQuantities<Ibis::dual> cq,FlowStates<Ibis::dual> fs);

    ~SteadyStateLinearisation() {}

    // SystemLinearisation interface
    void matrix_vector_product(Ibis::Vector<Ibis::real>& vec,
                               Ibis::Vector<Ibis::real>& result);

    void eval_rhs();

    KOKKOS_INLINE_FUNCTION
    Ibis::real& rhs(const size_t i) const { return rhs_(i); }

    KOKKOS_INLINE_FUNCTION
    Ibis::real& rhs(const size_t cell_i, const size_t cons_i) const {
        return rhs_(cell_i * n_cons_ + cons_i);
    }

    Ibis::Vector<Ibis::real>& rhs() { return rhs_; }

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
    ConservedQuantities<Ibis::dual> cq_;  // the current solution (not owned)
    FlowStates<Ibis::dual> fs_;  // the current solution (not owned)

    Ibis::Vector<Ibis::real> rhs_;  // the rhs of the system of equations

    ConservedQuantities<Ibis::dual> residuals_;  // temporary storage for residuals
    FlowStates<Ibis::dual> fs_tmp_;  // temporary storage for perturbed flow states
    ConservedQuantities<Ibis::dual> cq_tmp_;  // storage for perturbed cq

    // the simulation
    std::shared_ptr<Sim<Ibis::dual>> sim_;
};

class SteadyState : public Solver {
public:
    SteadyState(json config, GridBlock<Ibis::dual>& grid, std::string grid_dir,
                std::string flow_dir);

    ~SteadyState() {}

    int solve();

private:
    // The Jfnk solver
    Jfnk jfnk_;

    // configuration
    unsigned int print_frequency_;
    unsigned int plot_frequency_;
    unsigned int diagnostics_frequency_;

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
    int max_step() const { return jfnk_.max_steps(); }
    int count_bad_cells() { return sim_->fv.count_bad_cells(fs_, sim_->grid.num_cells()); }

private:
    // memory
    ConservedQuantities<Ibis::dual> cq_;
    FlowStates<Ibis::dual> fs_;

    // the core simulation
    std::shared_ptr<Sim<Ibis::dual>> sim_;
};

#endif
