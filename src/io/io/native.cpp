#include <io/binary_util.h>
#include <io/native.h>
#include <spdlog/spdlog.h>

#include "gas/transport_properties.h"

template <typename T>
int NativeTextOutput<T>::write(const typename FlowStates<T>::mirror_type& fs,
                               FiniteVolume<T>& fv, const GridBlock<T>& grid,
                               const IdealGas<T>& gas_model,
                               const TransportProperties<T>& trans_prop,
                               std::string plot_dir, std::string time_dir,
                               Ibis::real time) {
    (void)gas_model;
    (void)trans_prop;
    (void)fv;
    std::string dir = plot_dir + "/" + time_dir;
    std::ofstream meta_f(dir + "/meta_data.json");
    json meta;
    meta["time"] = time;
    meta_f << meta.dump(4);
    meta_f.close();

    std::ofstream flows(plot_dir + "/flows", std::ios_base::app);
    flows << time_dir << std::endl;
    flows.close();

    std::ofstream temp(dir + "/T");
    if (!temp) {
        spdlog::error("failed to open {}", dir + "/T");
        return 1;
    }
    temp << std::fixed << std::setprecision(16);
    for (size_t cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        temp << fs.gas.temp(cell_i) << std::endl;
    }
    temp.close();

    std::ofstream pressure(dir + "/p");
    if (!pressure) {
        spdlog::error("failed to open {}", dir + "/p");
        return 1;
    }
    pressure << std::fixed << std::setprecision(16);
    for (size_t cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        pressure << fs.gas.pressure(cell_i) << std::endl;
    }
    pressure.close();

    std::ofstream vx(dir + "/vx");
    if (!vx) {
        spdlog::error("failed to open {}", dir + "/vx");
        return 1;
    }
    vx << std::fixed << std::setprecision(16);
    for (size_t cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        vx << fs.vel.x(cell_i) << std::endl;
    }
    vx.close();

    std::ofstream vy(dir + "/vy");
    if (!vy) {
        spdlog::error("failed to open {}", dir + "/vy");
        return 1;
    }
    vy << std::fixed << std::setprecision(16);
    for (size_t cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        vy << fs.vel.y(cell_i) << std::endl;
    }
    vy.close();

    if (grid.dim() == 3) {
        std::ofstream vz(dir + "/vz");
        if (!vz) {
            spdlog::error("failed to open {}", dir + "/vz");
            return 1;
        }
        vz << std::fixed << std::setprecision(16);
        for (size_t cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
            vz << fs.vel.z(cell_i) << std::endl;
        }
        vz.close();
    }

    return 0;
}
template class NativeTextOutput<Ibis::real>;

template <typename T>
int NativeTextInput<T>::read(typename FlowStates<T>::mirror_type& fs,
                             const GridBlock<T>& grid, const IdealGas<T>& gas_model,
                             const TransportProperties<T>& trans_prop, std::string dir,
                             json& meta_data) {
    (void)trans_prop;
    size_t num_cells = grid.num_cells();
    std::ifstream meta_f(dir + "/meta_data.json");
    if (!meta_f) {
        spdlog::error("Unable to load {}", dir + "/meta_dta.json");
        return 1;
    }
    meta_data = json::parse(meta_f);
    meta_f.close();

    std::string line;
    std::ifstream temp(dir + "/T");
    if (!temp) {
        spdlog::error("Unable to load {}", dir + "/T");
        return 1;
    }
    size_t cell_i = 0;
    while (getline(temp, line)) {
        fs.gas.temp(cell_i) = stod(line);
        cell_i++;
    }
    if (cell_i != num_cells) {
        spdlog::error(
            "Incorrect number of values in initial T. {} cells, but {} "
            "temperature values",
            num_cells, cell_i);
    }

    std::ifstream pressure(dir + "/p");
    if (!pressure) {
        spdlog::error("Unable to load {}", dir + "/p");
        return 1;
    }
    cell_i = 0;
    while (getline(pressure, line)) {
        fs.gas.pressure(cell_i) = stod(line);
        cell_i++;
    }
    pressure.close();

    std::ifstream vx(dir + "/vx");
    if (!vx) {
        spdlog::error("Unable to load {}", dir + "/vx");
        return 1;
    }
    cell_i = 0;
    while (getline(vx, line)) {
        fs.vel.x(cell_i) = stod(line);
        cell_i++;
    }
    vx.close();

    std::ifstream vy(dir + "/vy");
    if (!vy) {
        spdlog::error("Unable to load {}", dir + "/vy");
        return 1;
    }
    cell_i = 0;
    while (getline(vy, line)) {
        fs.vel.y(cell_i) = stod(line);
        cell_i++;
    }
    vy.close();

    if (grid.dim() == 3) {
        std::ifstream vz(dir + "/vz");
        if (!vz) {
            spdlog::error("Unable to load {}", dir + "/vz");
            return 1;
        }

        cell_i = 0;
        while (getline(vz, line)) {
            fs.vel.z(cell_i) = stod(line);
            cell_i++;
        }
        vz.close();
    }

    // gas_model.update_thermo_from_pT(fs.gas);

    for (size_t cell_i = 0; cell_i < num_cells; cell_i++) {
        gas_model.update_thermo_from_pT(fs.gas, cell_i);
    }

    return 0;
}
template class NativeTextInput<Ibis::real>;

template <typename T>
int NativeBinaryOutput<T>::write(const typename FlowStates<T>::mirror_type& fs,
                                 FiniteVolume<T>& fv, const GridBlock<T>& grid,
                                 const IdealGas<T>& gas_model,
                                 const TransportProperties<T>& trans_prop,
                                 std::string plot_dir, std::string time_dir,
                                 Ibis::real time) {
    (void)gas_model;
    (void)trans_prop;
    (void)fv;
    std::string dir = plot_dir + "/" + time_dir;
    std::ofstream meta_f(dir + "/meta_data.json");
    json meta;
    meta["time"] = time;
    meta_f << meta.dump(4);
    meta_f.close();

    std::ofstream flows(plot_dir + "/flows", std::ios_base::app);
    flows << time_dir << std::endl;
    flows.close();

    std::ofstream temp(dir + "/T", std::ios::binary);
    if (!temp) {
        spdlog::error("failed to open {}", dir + "/T");
        return 1;
    }
    for (size_t cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        write_binary<Ibis::real>(temp, fs.gas.temp(cell_i));
    }
    temp.close();

    std::ofstream pressure(dir + "/p", std::ios::binary);
    if (!pressure) {
        spdlog::error("failed to open {}", dir + "/p");
        return 1;
    }
    for (size_t cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        write_binary<Ibis::real>(pressure, fs.gas.pressure(cell_i));
    }
    pressure.close();

    std::ofstream vx(dir + "/vx", std::ios::binary);
    if (!vx) {
        spdlog::error("failed to open {}", dir + "/vx");
        return 1;
    }
    for (size_t cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        write_binary<Ibis::real>(vx, fs.vel.x(cell_i));
    }
    vx.close();

    std::ofstream vy(dir + "/vy", std::ios::binary);
    if (!vy) {
        spdlog::error("failed to open {}", dir + "/vy");
        return 1;
    }
    for (size_t cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
        write_binary<Ibis::real>(vy, fs.vel.y(cell_i));
    }
    vy.close();

    if (grid.dim() == 3) {
        std::ofstream vz(dir + "/vz", std::ios::binary);
        if (!vz) {
            spdlog::error("failed to open {}", dir + "/vz");
            return 1;
        }
        for (size_t cell_i = 0; cell_i < grid.num_cells(); cell_i++) {
            write_binary<Ibis::real>(vz, fs.vel.z(cell_i));
        }
        vz.close();
    }

    return 0;
}
template class NativeBinaryOutput<Ibis::real>;

template <typename T>
int NativeBinaryInput<T>::read(typename FlowStates<T>::mirror_type& fs,
                               const GridBlock<T>& grid, const IdealGas<T>& gas_model,
                               const TransportProperties<T>& trans_prop, std::string dir,
                               json& meta_data) {
    (void)trans_prop;

    size_t num_cells = grid.num_cells();
    std::ifstream meta_f(dir + "/meta_data.json");
    if (!meta_f) {
        spdlog::error("Unable to load {}", dir + "/meta_dta.json");
        return 1;
    }
    meta_data = json::parse(meta_f);
    meta_f.close();

    std::string line;
    std::ifstream temp(dir + "/T", std::ios::binary);
    if (!temp) {
        spdlog::error("Unable to load {}", dir + "/T");
        return 1;
    }
    for (size_t cell_i = 0; cell_i < num_cells; cell_i++) {
        read_binary<Ibis::real>(temp, fs.gas.temp(cell_i));
    }

    std::ifstream pressure(dir + "/p", std::ios::binary);
    if (!pressure) {
        spdlog::error("Unable to load {}", dir + "/p");
        return 1;
    }
    for (size_t cell_i = 0; cell_i < num_cells; cell_i++) {
        read_binary<Ibis::real>(pressure, fs.gas.pressure(cell_i));
    }
    pressure.close();

    std::ifstream vx(dir + "/vx", std::ios::binary);
    if (!vx) {
        spdlog::error("Unable to load {}", dir + "/vx");
        return 1;
    }
    for (size_t cell_i = 0; cell_i < num_cells; cell_i++) {
        read_binary<Ibis::real>(vx, fs.vel.x(cell_i));
    }
    vx.close();

    std::ifstream vy(dir + "/vy", std::ios::binary);
    if (!vy) {
        spdlog::error("Unable to load {}", dir + "/vy");
        return 1;
    }
    for (size_t cell_i = 0; cell_i < num_cells; cell_i++) {
        read_binary<Ibis::real>(vy, fs.vel.y(cell_i));
    }
    vy.close();

    if (grid.dim() == 3) {
        std::ifstream vz(dir + "/vz", std::ios::binary);
        if (!vz) {
            spdlog::error("Unable to load {}", dir + "/vz");
            return 1;
        }

        for (size_t cell_i = 0; cell_i < num_cells; cell_i++) {
            read_binary<Ibis::real>(vz, fs.vel.z(cell_i));
        }
        vz.close();
    }

    // gas_model.update_thermo_from_pT(fs.gas);

    for (size_t cell_i = 0; cell_i < num_cells; cell_i++) {
        gas_model.update_thermo_from_pT(fs.gas, cell_i);
    }

    return 0;
}
template class NativeBinaryInput<Ibis::real>;
