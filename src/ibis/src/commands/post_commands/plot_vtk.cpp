#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include "plot_vtk.h"
#include "../../../src/grid/src/grid.h"
#include "../../../src/io/src/io.h"
#include "../../../src/gas/src/flow_state.h"
#include "../../config.h"


template <typename T>
void plot_vtk(json directories){
    std::string config_dir = directories.at("config_dir");
    std::string flow_dir = directories.at("flow_dir");
    std::string grid_dir = directories.at("grid_dir");
    std::string plot_dir = directories.at("plot_dir");

    // read the flows file to figure out what flow files exist
    std::ifstream flows(config_dir + "/flows");
    std::vector<std::string> dirs;
    std::string line;
    while (std::getline(flows, line)){
        dirs.push_back(line);
    }
    json config = read_config(directories);

    FVIO<T> io(FlowFormat::Native, FlowFormat::Vtk, flow_dir, plot_dir);
    GridBlock<T> grid(grid_dir+"/block_0000.su2", config.at("grid"));
    FlowStates<T> fs(grid.num_cells());
    for (unsigned int time_idx = 0; time_idx < dirs.size(); time_idx++){
        io.read(fs, grid, time_idx);
        io.write(fs, grid, 0.0);
    }
    io.write_coordinating_file();
}
template void plot_vtk<double>(json);
