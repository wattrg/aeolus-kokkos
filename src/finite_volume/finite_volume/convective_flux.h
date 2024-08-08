#ifndef CONVECTIVE_FLUX_H
#define CONVECTIVE_FLUX_H

#include <finite_volume/flux_calc.h>
#include <finite_volume/gradient.h>
#include <finite_volume/limiter.h>
#include <gas/flow_state.h>

#include <nlohmann/json.hpp>

#include "grid/grid.h"

using json = nlohmann::json;

enum class ThermoReconstructionVars {
    rho_p,
    rho_T,
    rho_u,
    p_T,
};

struct RequiredGradients {
    RequiredGradients() {}

    bool pressure = false;
    bool temp = false;
    bool u = false;
    bool rho = false;
};

template <typename T>
class ConvectiveFlux {
public:
    ConvectiveFlux() {}

    ConvectiveFlux(const GridBlock<T>& grid, json config);

    // Compute the convective fluxes. Includes gradient calculation,
    // but not boundary conditions
    void compute_convective_flux(const FlowStates<T>& flow_states,
                                 const GridBlock<T>& grid, IdealGas<T>& gas_model,
                                 Gradients<T>& cell_grad, WLSGradient<T>& grad_calc,
                                 ConservedQuantities<T>& flux, bool allow_reconstruction);

    // Compute the convective gradients. This could be private,
    // except for the fact that we might want to compute
    // the gradients in post without computing the actual flux
    // values afterwards.
    void compute_convective_gradient(const FlowStates<T>& flow_states,
                                     const GridBlock<T>& grid, Gradients<T>& cell_grad,
                                     WLSGradient<T>& grad_calc);

    void copy_reconstruct(const FlowStates<T>& flow_states, const GridBlock<T>& grid);

    void linear_reconstruct(const FlowStates<T>& flow_states, const GridBlock<T>& grid,
                            Gradients<T>& cell_grad, WLSGradient<T>& grad_calc,
                            IdealGas<T>& gas_model);

    void compute_limiters(const FlowStates<T>& flow_states, const GridBlock<T>& grid,
                          Gradients<T>& cell_grad);

    size_t reconstruction_order() const { return reconstruction_order_; }

    ThermoReconstructionVars thermo_interp() const { return reconstruction_vars_; }

    const RequiredGradients required_gradients() const;

private:
    // The flow states to the left of the interface
    FlowStates<T> left_;

    // The flow states to the right of the interface
    FlowStates<T> right_;

    // The flux calculator
    std::unique_ptr<FluxCalculator<T>> flux_calculator_;

    // reconstruction order
    size_t reconstruction_order_;

    ThermoReconstructionVars reconstruction_vars_;

    // The limiter
    std::unique_ptr<Limiter<T>> limiter_;

    // Storage for the limiter values
    LimiterValues<T> limiters_;
};

#endif
