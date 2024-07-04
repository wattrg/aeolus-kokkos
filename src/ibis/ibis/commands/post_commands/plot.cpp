
#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <grid/grid.h>
#include <ibis/commands/post_commands/plot.h>
#include <ibis/config.h>
#include <io/io.h>
#include <spdlog/spdlog.h>

#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "finite_volume/finite_volume.h"

int plot(FlowFormat format, std::vector<std::string> extras, int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    json directories = read_directories();
    if (format == FlowFormat::VtkBinary) {
        plot_vtk<Ibis::real, true>(directories, extras);
    } else if (format == FlowFormat::VtkText) {
        plot_vtk<Ibis::real, false>(directories, extras);
    } else if (format == FlowFormat::NativeText || format == FlowFormat::NativeBinary) {
        spdlog::error("Unable to plot in Native format");
        throw std::runtime_error("Unable to plot in Native format");
    }
    Kokkos::finalize();

    return 0;
}

template <typename T, bool binary>
void plot_vtk(json directories, std::vector<std::string> extra_vars) {
    std::string flow_dir = directories.at("flow_dir");
    std::string grid_dir = directories.at("grid_dir");
    std::string plot_dir = directories.at("plot_dir");

    // read the flows file to figure out what flow files exist
    std::ifstream flows(flow_dir + "/flows");
    std::vector<std::string> dirs;
    std::string line;
    while (std::getline(flows, line)) {
        dirs.push_back(line);
    }
    json config = read_config(directories);
    IdealGas<Ibis::real> gas_model{config.at("gas_model")};
    TransportProperties<Ibis::real> trans_prop{config.at("transport_properties")};

    // get the input and output flow formats
    FlowFormat flow_format = string_to_flow_format(config.at("io").at("flow_format"));
    constexpr FlowFormat plot_format =
        (binary) ? FlowFormat::VtkBinary : FlowFormat::VtkText;
    FVIO<T> io(flow_format, plot_format, flow_dir, plot_dir);

    for (auto& extra_var : extra_vars) {
        io.add_output_variable(extra_var);
    }

    GridBlock<T> grid(grid_dir + "/block_0000.su2", config.at("grid"));
    FiniteVolume<T> fv(grid, config);
    FlowStates<T> fs(grid.num_total_cells());
    for (unsigned int time_idx = 0; time_idx < dirs.size(); time_idx++) {
        json meta_data;
        io.read(fs, grid, gas_model, trans_prop, meta_data, time_idx);
        io.write(fs, fv, grid, gas_model, trans_prop, meta_data.at("time"));
        spdlog::info("Written VTK file at time index {}", time_idx);
    }
    io.write_coordinating_file();
}
template void plot_vtk<Ibis::real, false>(json, std::vector<std::string>);
template void plot_vtk<Ibis::real, true>(json, std::vector<std::string>);
