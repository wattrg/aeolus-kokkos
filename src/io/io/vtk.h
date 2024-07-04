#ifndef VTK_FLOW_FORMAT_H
#define VTK_FLOW_FORMAT_H

#include <io/accessor.h>
#include <io/io.h>

#include "finite_volume/finite_volume.h"
#include "gas/transport_properties.h"

template <typename T>
class VtkTextOutput : public FVOutput<T> {
public:
    VtkTextOutput();

    int write(const typename FlowStates<T>::mirror_type& fs, FiniteVolume<T>& fv,
              const GridBlock<T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop, std::string plot_dir,
              std::string time_dir, Ibis::real time);

    void write_coordinating_file(std::string plot_dir);

private:
    std::vector<Ibis::real> times_;
    std::vector<std::string> dirs_;
};

template <typename T>
class VtkBinaryOutput : public FVOutput<T> {
public:
    VtkBinaryOutput();

    int write(const typename FlowStates<T>::mirror_type& fs, FiniteVolume<T>& fv,
              const GridBlock<T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop, std::string plot_dir,
              std::string time_dir, Ibis::real time);

    void write_coordinating_file(std::string plot_dir);

private:
    std::vector<Ibis::real> times_;
    std::vector<std::string> dirs_;

    std::vector<std::byte> packed_data_;

private:
    void write_scalar_field_binary(
        std::ofstream& f, const FlowStates<T, array_layout, host_mem_space> fs,
        FiniteVolume<T>& fv, const GridBlock<T, host_exec_space, array_layout>& grid,
        std::shared_ptr<ScalarAccessor<T>> accessor, const IdealGas<T>& gas_model,
        std::string name, std::string type, size_t num_values);

    void write_vector_field_binary(
        std::ofstream& f, const FlowStates<T, array_layout, host_mem_space> fs,
        FiniteVolume<T>& fv, const GridBlock<T, host_exec_space, array_layout>& grid,
        std::shared_ptr<VectorAccessor<T>> accessor, const IdealGas<T>& gas_model,
        std::string name, std::string type, size_t num_values);

    void write_int_view_binary(
        std::ofstream& f, const Kokkos::View<size_t*, array_layout, host_mem_space>& view,
        std::string name, std::string type, bool skip_first = false);

    void write_elem_type_binary(
        std::ofstream& f, const Field<ElemType, array_layout, host_mem_space>& types);

    void write_vector3s_binary(std::ofstream& f,
                               const Vector3s<T, array_layout, host_mem_space>& vec,
                               std::string name, std::string type, size_t num_values);

    void write_appended_data(std::ofstream& f);
};

#endif
