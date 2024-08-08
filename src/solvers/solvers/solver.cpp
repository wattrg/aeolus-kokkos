#include <solvers/runge_kutta.h>
#include <solvers/solver.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>

#include "solvers/steady_state.h"

using json = nlohmann::json;

Solver::Solver(std::string grid_dir, std::string flow_dir)
    : grid_dir_(grid_dir), flow_dir_(flow_dir) {}

int Solver::solve() {
    int success = initialise();
    if (success != 0) {
        spdlog::error("Failed to initialise runge kutta solver");
        return success;
    }
    spdlog::stopwatch sw;
    for (size_t step = 0; step < max_step(); step++) {
        int result = take_step();

        if (residuals_this_step(step)) {
            write_residuals(step);
        }

        if (result != 0) {
            spdlog::error("step {} failed", step);
            plot_solution(step);
            return 1;
        }

        int bad_cells = count_bad_cells();
        if (bad_cells > 0) {
            spdlog::error("Encountered {} bad cells on step {}", bad_cells, step);
            plot_solution(step);
            return 1;
        }

        if (stop_now(step)) {
            std::string reason = stop_reason(step);
            spdlog::info("STOPPING: {}", reason);
            plot_solution(step);
            break;
        }

        if (print_this_step(step)) {
            print_progress(step, sw.elapsed().count());
        }


        if (plot_this_step(step)) {
            plot_solution(step);
        }
    }
    spdlog::info("Elapsed Wall Clock: {:.3}s", sw);
    finalise();

    return 0;
}

std::unique_ptr<Solver> make_solver(json config, std::string grid_dir,
                                    std::string flow_dir) {
    std::string grid_file = grid_dir + "/block_0000.su2";
    json solver_config = config.at("solver");
    json grid_config = config.at("grid");
    std::string solver_name = solver_config.at("name");
    if (solver_name == "runge_kutta") {
        GridBlock<Ibis::real> grid(grid_file, grid_config);
        return std::unique_ptr<Solver>(
            new RungeKutta(config, std::move(grid), grid_dir, flow_dir));
    } else if (solver_name == "steady_state") {
        GridBlock<Ibis::dual> grid(grid_file, grid_config);
        return std::unique_ptr<Solver>(
            new SteadyState(config, std::move(grid), grid_dir, flow_dir));
    } else {
        spdlog::error("Unknown solver {}", solver_name);
        throw new std::runtime_error("Unknown solver");
    }
    return NULL;
}
