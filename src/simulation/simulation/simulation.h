#ifndef SIMULATION_H
#define SIMULATION_H

#include <finite_volume/finite_volume.h>
#include <gas/gas_model.h>
#include <gas/transport_properties.h>
#include <grid/grid.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

template <typename T>
struct Sim {
    Sim(GridBlock<T>&& grid, json config);

    FiniteVolume<T> fv;
    GridBlock<T> grid;
    IdealGas<T> gas_model;
    TransportProperties<T> trans_prop;
};

#endif
