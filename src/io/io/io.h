#ifndef FV_IO_H
#define FV_IO_H

#include <gas/flow_state.h>
#include <gas/gas_model.h>
#include <grid/grid.h>
#include <io/accessor.h>
#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>

#include "finite_volume/finite_volume.h"
#include "gas/transport_properties.h"

using json = nlohmann::json;

enum class FlowFormat { NativeText, NativeBinary, VtkText, VtkBinary };

FlowFormat string_to_flow_format(std::string format);

template <typename T>
class FVInput {
public:
    virtual ~FVInput() {}

    virtual int read(typename FlowStates<T>::mirror_type& fs, const GridBlock<T>& grid,
                     const IdealGas<T>& gas_model,
                     const TransportProperties<T>& trans_prop, std::string dir,
                     json& meta_data) = 0;
};

template <typename T>
class FVOutput {
public:
    virtual ~FVOutput() {}

    virtual int write(const typename FlowStates<T>::mirror_type& fs, FiniteVolume<T>& fv,
                      const GridBlock<T>& grid, const IdealGas<T>& gas_model,
                      const TransportProperties<T>& trans_prop, std::string plot_dir,
                      std::string time_dir, Ibis::real time) = 0;

    void add_variable(std::string name);

    virtual void write_coordinating_file(std::string plot_dir) = 0;

protected:
    std::map<std::string, std::shared_ptr<ScalarAccessor<T>>> m_scalar_accessors;
    std::map<std::string, std::shared_ptr<VectorAccessor<T>>> m_vector_accessors;
};

template <typename T>
class FVIO {
public:
    // constructors
    FVIO(FlowFormat input_format, FlowFormat output_format, std::string input_dir,
         std::string output_dir, int time_index);

    FVIO(FlowFormat input, FlowFormat output);

    FVIO(FlowFormat input, FlowFormat output, std::string input_dir,
         std::string output_dir);

    FVIO(FlowFormat input, FlowFormat output, int time_index);

    FVIO(int time_index);

    FVIO();

    // read a flow state
    int read(FlowStates<T>& fs, const GridBlock<T>& grid, const IdealGas<T>& gas_model,
             const TransportProperties<T>& trans_prop, json& meta_data, int time_idx);

    // write a flow state
    int write(const FlowStates<T>& flow_state, FiniteVolume<T>& fv,
              const GridBlock<T>& grid, const IdealGas<T>& gas_model,
              const TransportProperties<T>& trans_prop, Ibis::real time);

    void add_output_variable(std::string name) { output_->add_variable(name); }

    void write_coordinating_file();

private:
    std::unique_ptr<FVInput<T>> input_;
    std::unique_ptr<FVOutput<T>> output_;
    int time_index_;
    std::string input_dir_;
    std::string output_dir_;
};

#endif
