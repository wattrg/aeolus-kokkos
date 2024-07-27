#include <finite_volume/finite_volume.h>
#include <gas/gas_model.h>
#include <gas/transport_properties.h>
#include <simulation/simulation.h>
#include <util/numeric_types.h>

template <typename T>
Sim<T>::Sim(GridBlock<T>&& grid_, json config) : grid(grid_) {
    fv = FiniteVolume<T>(grid_, config.at("finite_volume"));
    gas_model = IdealGas<T>(config.at("gas_model"));
    trans_prop = TransportProperties<T>(config.at("transport_properties"));
}

template struct Sim<Ibis::real>;
template struct Sim<Ibis::dual>;
