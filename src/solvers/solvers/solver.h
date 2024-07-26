#ifndef SOLVER_H
#define SOLVER_H

// #include <finite_volume/conserved_quantities.h>
// #include <finite_volume/finite_volume.h>
#include <gas/flow_state.h>
#include <grid/grid.h>
#include <string.h>
// #include <util/types.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Solver {
public:
    Solver(std::string grid_dir, std::string flow_dir);
    int solve();
    virtual ~Solver() {}

protected:
    int write_solution();

    // the main parts of the solver
    virtual int initialise() = 0;
    virtual int finalise() = 0;
    virtual int take_step() = 0;

    // some io utilities
    virtual bool print_this_step(unsigned int step) = 0;
    virtual bool plot_this_step(unsigned int step) = 0;
    virtual bool residuals_this_step(unsigned int step) = 0;
    virtual bool write_residuals(unsigned int step) = 0;
    virtual int plot_solution(unsigned int step) = 0;
    virtual void print_progress(unsigned int step, Ibis::real wc) = 0;
    virtual bool stop_now(unsigned int step) = 0;
    virtual std::string stop_reason(unsigned step) = 0;
    virtual int max_step() const = 0;

    // error checking
    virtual int count_bad_cells() = 0;

    // book keeping
    unsigned int max_step_ = 0;
    std::string grid_dir_;
    std::string flow_dir_;
};

std::unique_ptr<Solver> make_solver(json config, std::string grid_dir,
                                    std::string flow_dir);

template <typename T>
int read_initial_condition(FlowStates<T>& fs, std::string flow_dir, int num_cells);

template <typename T>
int write_flow_solution(const FlowStates<T>& fs, const GridBlock<T>& grid,
                        const std::string flow_dir, int flow_i);

#endif
