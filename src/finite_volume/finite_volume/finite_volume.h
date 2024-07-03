#ifndef FINITE_VOLUME_H
#define FINITE_VOLUME_H

#include <finite_volume/boundaries/boundary.h>
#include <finite_volume/conserved_quantities.h>
#include <finite_volume/convective_flux.h>
#include <finite_volume/flux_calc.h>
#include <finite_volume/gradient.h>
#include <finite_volume/limiter.h>
#include <finite_volume/viscous_flux.h>
#include <gas/flow_state.h>
#include <gas/gas_model.h>
#include <gas/transport_properties.h>
#include <grid/grid.h>
#include <spdlog/spdlog.h>
#include <util/numeric_types.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

/**
 * Handles the fluid dynamics/finite volume related things
 *
 * @tparam T The type of number flow variables are stored with
 */
template <typename T>
class FiniteVolume {
public:
    FiniteVolume() {}

    FiniteVolume(const GridBlock<T>& grid, json config);

    /**
     * Compute the time derivative of a particular flow state
     *
     * @param[in] flow_state The flow state to compute the time derivative for
     * @param[in] grid The grid
     * @param[out] dudt The time derivatives of conserved quantities
     * @param[in] gas_model The gas model
     * @param[in] trans_prop The transport properties
     */
    size_t compute_dudt(FlowStates<T>& flow_state, const GridBlock<T>& grid,
                        ConservedQuantities<T>& dudt, IdealGas<T>& gas_model,
                        TransportProperties<T>& trans_prop);

    /**
     * Estimate the allowable global time step for a given flow
     * state on a given grid
     *
     * @param flow_state The flow state to estimate the allowable
     *     time step for
     * @param grid The grid to compute the time step for
     * @param gas_model The gas model
     * @param trans_prop The transport properties
     * @return the size of the time step
     */
    Ibis::real estimate_dt(const FlowStates<T>& flow_state, GridBlock<T>& grid,
                           IdealGas<T>& gas_model, TransportProperties<T>& trans_prop);

    // methods
    // these have to be public for NVCC, but they shouldn't really need to
    // be accessed from outside of the class. Although sometimes the
    // post-processing will call them.

    // Apply pre-reconstruction boundary conditions
    void apply_pre_reconstruction_bc(FlowStates<T>& fs, const GridBlock<T>& grid,
                                     const IdealGas<T>& gas_model,
                                     const TransportProperties<T>& trans_prop);

    // Apply pre-reconstruction boundary conditions
    void apply_pre_viscous_grad_bc(FlowStates<T>& fs, const GridBlock<T>& grid,
                                   const IdealGas<T>& gas_model,
                                   const TransportProperties<T>& trans_prop);

    // Perform the surface integral of fluxes over the cells
    void flux_surface_integral(const GridBlock<T>& grid, ConservedQuantities<T>& dudt);

    // Count the number of bad cells in the domain
    size_t count_bad_cells(const FlowStates<T>& fs, const size_t num_cells);

public:
    // methods for IO
    const Gradients<T>& cell_gradients() const { return cell_grad_; }

    // viscous gradients for post-processing
    void compute_viscous_gradient(FlowStates<T>& fs, const GridBlock<T>& grid,
                                  const IdealGas<T>& gas_model,
                                  const TransportProperties<T>& trans_prop);

    // convective gradients for post-processing
    void compute_convective_gradient(FlowStates<T>& fs, const GridBlock<T>& grid,
                                     const IdealGas<T>& gas_model,
                                     const TransportProperties<T>& trans_prop);

private:
    // The flux values
    ConservedQuantities<T> flux_;

    // The convective flux calculator
    ConvectiveFlux<T> convective_flux_;

    // The viscous flux calculator
    ViscousFlux<T> viscous_flux_;

    // boundary conditions
    std::vector<std::shared_ptr<BoundaryCondition<T>>> bcs_{};

    // The interfaces on each boundary
    std::vector<Field<size_t>> bc_interfaces_{};

    // number of spatial dimensions
    size_t dim_;

    // Gradient calculator
    WLSGradient<T> grad_calc_;

    // Storage for gradients at cells
    Gradients<T> cell_grad_;
};

#endif
