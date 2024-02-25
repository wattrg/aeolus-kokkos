#include <grid/grid.h>
#include <ibis/commands/run/run.h>
#include <ibis/config.h>
#include <ibis_version_info.h>
#include <solvers/solver.h>
#include <spdlog/spdlog.h>

#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void print_header() {
    spdlog::info("ibis - cfd solver");
    spdlog::info("git branch: {}", Ibis::GIT_BRANCH);
    if (Ibis::GIT_CLEAN_STATUS == "clean") {
        spdlog::info("git commit: {}", Ibis::GIT_COMMIT_HASH);
    } else {
        spdlog::info("git commit: {}-dirty", Ibis::GIT_COMMIT_HASH);
    }
    spdlog::info("revision date: {}", Ibis::GIT_COMMIT_DATE);
    spdlog::info("build date: {}", Ibis::IBIS_BUILD_DATE);
}

void print_config_info(json config) {
    spdlog::info("solver: {}", std::string(config.at("solver").at("name")));
    spdlog::info(
        "flux calculator: {}",
        std::string(config.at("convective_flux").at("flux_calculator")));
}

int run(int argc, char* argv[]) {
    json directories = read_directories();
    json config = read_config(directories);

    print_header();
    print_config_info(config);

    std::string grid_dir = directories.at("grid_dir");
    std::string flow_dir = directories.at("flow_dir");
    Kokkos::initialize(argc, argv);
    int result;

    {
        // we need to make the solver (and thus allocate all the kokkos memory)
        // inside a block, so that the solver (and thus all kokkos managed
        // memory) is removed before Kokkos::finalise is called
        std::unique_ptr<Solver> solver =
            make_solver(config, grid_dir, flow_dir);
        result = solver->solve();
    }

    Kokkos::finalize();

    if (result != 0) {
        spdlog::error("run failed");
    } else {
        spdlog::info("run complete");
    }
    return result;
}
