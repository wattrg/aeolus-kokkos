
#include <finite_volume/finite_volume.h>
#include <io/accessor.h>
#include <io/io.h>
#include <io/native.h>
#include <io/vtk.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "gas/transport_properties.h"

FlowFormat string_to_flow_format(std::string format) {
    if (format == "native_text") {
        return FlowFormat::NativeText;
    } else if (format == "native_binary") {
        return FlowFormat::NativeBinary;
    } else if (format == "vtk_text") {
        return FlowFormat::VtkText;
    } else if (format == "vtk_binary") {
        return FlowFormat::VtkBinary;
    } else {
        spdlog::error("Unknown flow format {}", format);
        throw std::runtime_error("Unknown flow format");
    }
}

std::string pad_time_index(int time_idx, unsigned long len) {
    std::string time_index = std::to_string(time_idx);
    unsigned long extra_chars = len - std::min<unsigned long>(len, time_index.length());
    std::string padded_str = std::string(extra_chars, '0');
    return padded_str + time_index;
}

template <typename T>
std::unique_ptr<FVInput<T>> make_fv_input(FlowFormat format) {
    switch (format) {
        case FlowFormat::NativeText:
            return std::unique_ptr<FVInput<T>>(new NativeTextInput<T>());
        case FlowFormat::NativeBinary:
            return std::unique_ptr<FVInput<T>>(new NativeBinaryInput<T>());
        case FlowFormat::VtkText:
        case FlowFormat::VtkBinary:
            spdlog::error("Reading VTK files not supported");
            throw std::runtime_error("Reading VTK files not supported");
        default:
            throw std::runtime_error("Unreachable");
    }
}

template <typename T>
std::unique_ptr<FVOutput<T>> make_fv_output(FlowFormat format) {
    switch (format) {
        case FlowFormat::NativeText:
            return std::unique_ptr<FVOutput<T>>(new NativeTextOutput<T>());
        case FlowFormat::NativeBinary:
            return std::unique_ptr<FVOutput<T>>(new NativeBinaryOutput<T>());
        case FlowFormat::VtkText:
            return std::unique_ptr<FVOutput<T>>(new VtkTextOutput<T>());
        case FlowFormat::VtkBinary:
            return std::unique_ptr<FVOutput<T>>(new VtkBinaryOutput<T>());
        default:
            throw std::runtime_error("Unreachable");
    }
}

template <typename T>
FVIO<T>::FVIO(FlowFormat input_format, FlowFormat output_format, std::string input_dir,
              std::string output_dir, int time_index)
    : input_(make_fv_input<T>(input_format)),
      output_(make_fv_output<T>(output_format)),
      time_index_(time_index),
      input_dir_(input_dir),
      output_dir_(output_dir) {}

template <typename T>
FVIO<T>::FVIO(FlowFormat input, FlowFormat output)
    : FVIO(input, output, "flow", "flow", 0) {}

template <typename T>
FVIO<T>::FVIO(FlowFormat input, FlowFormat output, std::string input_dir,
              std::string output_dir)
    : FVIO(input, output, input_dir, output_dir, 0) {}

template <typename T>
FVIO<T>::FVIO()
    : FVIO(FlowFormat::NativeBinary, FlowFormat::NativeBinary, "flow", "flow", 0) {}

template <typename T>
FVIO<T>::FVIO(int time_index)
    : FVIO(FlowFormat::NativeBinary, FlowFormat::NativeBinary, "flow", "flow",
           time_index) {}

template <typename T>
FVIO<T>::FVIO(FlowFormat input, FlowFormat output, int time_index)
    : FVIO(input, output, "flow", "flow", time_index) {}

template <typename T>
int FVIO<T>::write(const FlowStates<T>& fs, FiniteVolume<T>& fv, const GridBlock<T>& grid,
                   const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop,
                   Ibis::real time) {
    // get a copy of the flow states on the CPU
    auto fs_host = fs.host_mirror();
    fs_host.deep_copy(fs);

    std::string time_index = pad_time_index(time_index_, 4);
    std::string directory_name = output_dir_ + "/" + time_index;
    std::filesystem::create_directory(output_dir_);
    std::filesystem::create_directory(directory_name);
    int result = output_->write(fs_host, fv, grid, gas_model, trans_prop, output_dir_,
                                time_index, time);
    time_index_++;
    return result;
}

template <typename T>
int FVIO<T>::read(FlowStates<T>& fs, const GridBlock<T>& grid,
                  const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop,
                  json& meta_data, int time_idx) {
    // auto grid_host = grid.host_mirror();
    auto fs_host = fs.host_mirror();
    std::string time_index = pad_time_index(time_idx, 4);
    std::string directory_name = input_dir_ + "/" + time_index;
    int result =
        input_->read(fs_host, grid, gas_model, trans_prop, directory_name, meta_data);
    fs.deep_copy(fs_host);
    return result;
}

template <typename T>
void FVIO<T>::write_coordinating_file() {
    output_->write_coordinating_file(output_dir_);
}

template <typename T>
void FVOutput<T>::add_variable(std::string name) {
    if (name == "viscous_grad_vx") {
        this->m_vector_accessors.insert(
            {name, std::shared_ptr<VectorAccessor<T>>(new ViscousGradVxAccess<T>())});
    } else if (name == "viscous_grad_vy") {
        this->m_vector_accessors.insert(
            {name, std::shared_ptr<VectorAccessor<T>>(new ViscousGradVyAccess<T>())});
    } else if (name == "viscous_grad_vz") {
        this->m_vector_accessors.insert(
            {name, std::shared_ptr<VectorAccessor<T>>(new ViscousGradVzAccess<T>())});
    } else if (name == "convective_grad_vx") {
        this->m_vector_accessors.insert(
            {name, std::shared_ptr<VectorAccessor<T>>(new ConvectiveGradVxAccess<T>())});
    } else if (name == "convective_grad_vy") {
        this->m_vector_accessors.insert(
            {name, std::shared_ptr<VectorAccessor<T>>(new ConvectiveGradVyAccess<T>())});
    } else if (name == "convective_grad_vz") {
        this->m_vector_accessors.insert(
            {name, std::shared_ptr<VectorAccessor<T>>(new ConvectiveGradVzAccess<T>())});
    } else if (name == "viscous_grad_v") {
        this->m_vector_accessors.insert(
            {"viscous_grad_vx",
             std::shared_ptr<VectorAccessor<T>>(new ViscousGradVxAccess<T>())});
        this->m_vector_accessors.insert(
            {"viscous_grad_vy",
             std::shared_ptr<VectorAccessor<T>>(new ViscousGradVyAccess<T>())});
        this->m_vector_accessors.insert(
            {"viscous_grad_vz",
             std::shared_ptr<VectorAccessor<T>>(new ViscousGradVzAccess<T>())});
    } else if (name == "convective_grad_v") {
        this->m_vector_accessors.insert(
            {"convective_grad_vx",
             std::shared_ptr<VectorAccessor<T>>(new ConvectiveGradVxAccess<T>())});
        this->m_vector_accessors.insert(
            {"convective_grad_vy",
             std::shared_ptr<VectorAccessor<T>>(new ConvectiveGradVyAccess<T>())});
        this->m_vector_accessors.insert(
            {"convective_grad_vz",
             std::shared_ptr<VectorAccessor<T>>(new ConvectiveGradVzAccess<T>())});
    } else if (name == "cell_centre") {
        this->m_vector_accessors.insert(
            {"cell_centre",
             std::shared_ptr<CellCentreAccess<T>>(new CellCentreAccess<T>())});
    } else {
        spdlog::error("Unknown post-processing variable {}", name);
        throw std::runtime_error("Unknown post-process variable");
    }
}

template class FVIO<Ibis::real>;
template class FVIO<Ibis::dual>;
