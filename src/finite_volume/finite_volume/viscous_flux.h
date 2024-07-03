#ifndef VISCOUS_FLUX_H
#define VISCOUS_FLUX_H

#include <finite_volume/conserved_quantities.h>
#include <finite_volume/gradient.h>
#include <gas/flow_state.h>
#include <gas/gas_model.h>
#include <gas/transport_properties.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

template <typename T>
class ViscousFlux {
public:
    ViscousFlux() {}

    ViscousFlux(const GridBlock<T>& grid, json config);

    bool enabled() const { return enabled_; }

    void compute_viscous_gradient(const FlowStates<T>& flow_states,
                                  const GridBlock<T>& grid, Gradients<T>& cell_grad,
                                  WLSGradient<T>& grad_calc);

    void compute_viscous_flux(const FlowStates<T>& flow_states, const GridBlock<T>& grid,
                              const IdealGas<T>& gas_model,
                              const TransportProperties<T>& trans_prop,
                              Gradients<T>& cell_grad, WLSGradient<T>& grad_calc,
                              ConservedQuantities<T>& flux);

    // void compute_viscous_properties_at_faces(const FlowStates<T>& flow_states,
    //                                          const GridBlock<T>& grid,
    //                                          const IdealGas<T>& gas_model,
    //                                          Gradients<T>& cell_grad);

    const FlowStates<T>& face_fs() const { return face_fs_; }

    Ibis::real signal_factor() const { return signal_factor_; }

private:
    bool enabled_;
    FlowStates<T> face_fs_;
    Ibis::real signal_factor_;
};

#endif
