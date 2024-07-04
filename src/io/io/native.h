#ifndef NATIVE_FLOW_FORMAT_H
#define NATIVE_FLOW_FORMAT_H

#include <io/io.h>

#include "gas/transport_properties.h"

template <typename T>
class NativeTextInput : public FVInput<T> {
public:
    NativeTextInput() {}

    int read(typename FlowStates<T>::mirror_type& fs, const GridBlock<T>& grid,
             const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop,
             std::string dir, json& meta_data);
};

template <typename T>
class NativeTextOutput : public FVOutput<T> {
public:
    NativeTextOutput() {}

    int write(const typename FlowStates<T>::mirror_type& fs, FiniteVolume<T>& fv,
              const GridBlock<T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop, std::string plot_dir,
              std::string time_dir, Ibis::real time);

    void write_coordinating_file(std::string plot_dir) { (void)plot_dir; }
};

template <typename T>
class NativeBinaryInput : public FVInput<T> {
public:
    NativeBinaryInput() {}

    int read(typename FlowStates<T>::mirror_type& fs, const GridBlock<T>& grid,
             const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop,
             std::string dir, json& meta_data);
};

template <typename T>
class NativeBinaryOutput : public FVOutput<T> {
public:
    NativeBinaryOutput() {}

    int write(const typename FlowStates<T>::mirror_type& fs, FiniteVolume<T>& fv,
              const GridBlock<T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop, std::string plot_dir,
              std::string time_dir, Ibis::real time);

    void write_coordinating_file(std::string plot_dir) { (void)plot_dir; }
};

#endif
