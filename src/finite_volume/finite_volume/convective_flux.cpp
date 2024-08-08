#include <finite_volume/convective_flux.h>
#include <finite_volume/gradient.h>
#include <spdlog/spdlog.h>
#include <util/numeric_types.h>

#include <stdexcept>

std::string string_from_reconstruction_vars(ThermoReconstructionVars vars) {
    switch (vars) {
        case ThermoReconstructionVars::rho_p:
            return "rho_p";
        case ThermoReconstructionVars::rho_T:
            return "rho_T";
        case ThermoReconstructionVars::rho_u:
            return "rho_u";
        case ThermoReconstructionVars::p_T:
            return "p_T";
        default:
            throw new std::runtime_error("Shouldn't get here");
    }
}

ThermoReconstructionVars reconstruction_vars_from_string(std::string variables) {
    if (variables == "rho_p") {
        return ThermoReconstructionVars::rho_p;
    } else if (variables == "rho_T") {
        return ThermoReconstructionVars::rho_T;
    } else if (variables == "rho_u") {
        return ThermoReconstructionVars::rho_u;
    } else if (variables == "p_T") {
        return ThermoReconstructionVars::p_T;
    } else {
        spdlog::error("Unknown thermo interpolator {}", variables);
        throw new std::runtime_error("Unknown thermo interpolator");
    }
}

template <typename T>
const RequiredGradients ConvectiveFlux<T>::required_gradients() const {
    RequiredGradients required_grads = RequiredGradients();
    switch (reconstruction_vars_) {
        case ThermoReconstructionVars::rho_p:
            required_grads.rho = true;
            required_grads.pressure = true;
            break;
        case ThermoReconstructionVars::rho_T:
            required_grads.rho = true;
            required_grads.temp = true;
            break;
        case ThermoReconstructionVars::rho_u:
            required_grads.rho = true;
            required_grads.u = true;
            break;
        case ThermoReconstructionVars::p_T:
            required_grads.pressure = true;
            required_grads.u = true;
    }
    return required_grads;
}

template <typename T>
ConvectiveFlux<T>::ConvectiveFlux(const GridBlock<T>& grid, json config) {
    // allocate memory for the reconstructed states to the left and
    // right of the interfaces
    left_ = FlowStates<T>(grid.num_interfaces());
    right_ = FlowStates<T>(grid.num_interfaces());

    // set up the flux calculator
    flux_calculator_ = make_flux_calculator<T>(config.at("flux_calculator"));

    // set up reconstruction
    reconstruction_order_ = config.at("reconstruction_order");
    if (reconstruction_order_ > 1) {
        reconstruction_vars_ =
            reconstruction_vars_from_string(config.at("thermo_interpolator"));
        limiter_ = make_limiter<T>(config.at("limiter"));
        if (limiter_->enabled()) {
            const RequiredGradients grads = required_gradients();
            limiters_ = LimiterValues<T>(grid.num_cells(), grads.pressure, grads.rho,
                                         grads.temp, grads.u);
        }
    }
}

template <typename T>
void ConvectiveFlux<T>::compute_convective_flux(
    const FlowStates<T>& flow_states, const GridBlock<T>& grid, IdealGas<T>& gas_model,
    Gradients<T>& cell_grad, WLSGradient<T>& grad_calc, ConservedQuantities<T>& flux,
    bool allow_reconstruction) {
    // reconstruct
    int reconstruction_order = (allow_reconstruction) ? reconstruction_order_ : 1;
    switch (reconstruction_order) {
        case 1:
            copy_reconstruct(flow_states, grid);
            break;
        case 2:
            linear_reconstruct(flow_states, grid, cell_grad, grad_calc, gas_model);
            break;
        default:
            spdlog::error("Invalid reconstruction order {}", reconstruction_order_);
            throw new std::runtime_error("Invalid reconstruction order");
    }

    // rotate velocity to the interface reference frame
    Interfaces<T> faces = grid.interfaces();
    transform_to_local_frame(left_.vel, faces.norm(), faces.tan1(), faces.tan2());
    transform_to_local_frame(right_.vel, faces.norm(), faces.tan1(), faces.tan2());

    // compute the flux
    flux_calculator_->compute_flux(left_, right_, flux, gas_model, grid.dim() == 3);

    // rotate the fluxes to the global frame
    Vector3s<T> norm = faces.norm();
    Vector3s<T> tan1 = faces.tan1();
    Vector3s<T> tan2 = faces.tan2();
    Kokkos::parallel_for(
        "flux::transform_to_global", faces.size(), KOKKOS_LAMBDA(const size_t i) {
            T px = flux.momentum_x(i);
            T py = flux.momentum_y(i);
            T pz = 0.0;
            if (flux.dim() == 3) {
                pz = flux.momentum_z(i);
            }
            T x = px * norm.x(i) + py * tan1.x(i) + pz * tan2.x(i);
            T y = px * norm.y(i) + py * tan1.y(i) + pz * tan2.y(i);
            T z = px * norm.z(i) + py * tan1.z(i) + pz * tan2.z(i);
            flux.momentum_x(i) = x;
            flux.momentum_y(i) = y;
            if (flux.dim() == 3) {
                flux.momentum_z(i) = z;
            }
        });
}

template <typename T>
void ConvectiveFlux<T>::compute_convective_gradient(const FlowStates<T>& flow_states,
                                                    const GridBlock<T>& grid,
                                                    Gradients<T>& cell_grad,
                                                    WLSGradient<T>& grad_calc) {
    switch (reconstruction_vars_) {
        case ThermoReconstructionVars::rho_p:
            grad_calc.compute_gradients(grid, flow_states.gas.pressure(), cell_grad.p);
            grad_calc.compute_gradients(grid, flow_states.gas.rho(), cell_grad.rho);
            break;
        case ThermoReconstructionVars::rho_u:
            grad_calc.compute_gradients(grid, flow_states.gas.rho(), cell_grad.rho);
            grad_calc.compute_gradients(grid, flow_states.gas.energy(), cell_grad.u);
            break;
        case ThermoReconstructionVars::rho_T:
            grad_calc.compute_gradients(grid, flow_states.gas.rho(), cell_grad.rho);
            grad_calc.compute_gradients(grid, flow_states.gas.temp(), cell_grad.temp);
            break;
        case ThermoReconstructionVars::p_T:
            grad_calc.compute_gradients(grid, flow_states.gas.pressure(), cell_grad.p);
            grad_calc.compute_gradients(grid, flow_states.gas.temp(), cell_grad.temp);
            break;
    }

    grad_calc.compute_gradients(grid, flow_states.vel.x(), cell_grad.vx);
    grad_calc.compute_gradients(grid, flow_states.vel.y(), cell_grad.vy);
    if (grid.dim() == 3) {
        grad_calc.compute_gradients(grid, flow_states.vel.z(), cell_grad.vz);
    }
}

template <typename T>
void ConvectiveFlux<T>::copy_reconstruct(const FlowStates<T>& flow_states,
                                         const GridBlock<T>& grid) {
    size_t n_faces = grid.num_interfaces();
    FlowStates<T> this_left = left_;
    FlowStates<T> this_right = right_;
    Interfaces<T> interfaces = grid.interfaces();
    Kokkos::parallel_for(
        "Reconstruct", n_faces, KOKKOS_LAMBDA(const size_t i_face) {
            // copy left flow states
            size_t left = interfaces.left_cell(i_face);
            this_left.gas.temp(i_face) = flow_states.gas.temp(left);
            this_left.gas.pressure(i_face) = flow_states.gas.pressure(left);
            this_left.gas.rho(i_face) = flow_states.gas.rho(left);
            this_left.gas.energy(i_face) = flow_states.gas.energy(left);
            this_left.vel.x(i_face) = flow_states.vel.x(left);
            this_left.vel.y(i_face) = flow_states.vel.y(left);
            this_left.vel.z(i_face) = flow_states.vel.z(left);

            // copy right flow states
            size_t right = interfaces.right_cell(i_face);
            this_right.gas.temp(i_face) = flow_states.gas.temp(right);
            this_right.gas.pressure(i_face) = flow_states.gas.pressure(right);
            this_right.gas.rho(i_face) = flow_states.gas.rho(right);
            this_right.gas.energy(i_face) = flow_states.gas.energy(right);
            this_right.vel.x(i_face) = flow_states.vel.x(right);
            this_right.vel.y(i_face) = flow_states.vel.y(right);
            this_right.vel.z(i_face) = flow_states.vel.z(right);
        });
}

template <typename T>
KOKKOS_INLINE_FUNCTION T linear_interpolate(T value, Vector3s<T> grad, T dx, T dy, T dz,
                                            int i, T limiter, bool is_valid) {
    T grad_x = 0.0;
    T grad_y = 0.0;
    T grad_z = 0.0;
    if (is_valid) {
        grad_x = grad.x(i);
        grad_y = grad.y(i);
        grad_z = grad.z(i);
    }

    return value + limiter * (grad_x * dx + grad_y * dy + grad_z * dz);
}

template <typename T>
void ConvectiveFlux<T>::linear_reconstruct(const FlowStates<T>& flow_states,
                                           const GridBlock<T>& grid,
                                           Gradients<T>& cell_grad,
                                           WLSGradient<T>& grad_calc,
                                           IdealGas<T>& gas_model) {
    compute_convective_gradient(flow_states, grid, cell_grad, grad_calc);
    compute_limiters(flow_states, grid, cell_grad);

    auto limiters = limiters_;
    auto grad = cell_grad;
    auto cells = grid.cells();
    auto faces = grid.interfaces();
    auto left = left_;
    auto right = right_;
    size_t num_cells = grid.num_cells();
    bool limiter_enabled = limiter_->enabled();
    ThermoReconstructionVars thermo_interpolator = reconstruction_vars_;
    Kokkos::parallel_for(
        "FV::linear_reconstruct", grid.num_interfaces(), KOKKOS_LAMBDA(const int i_face) {
            // left state
            size_t left_cell = faces.left_cell(i_face);
            bool left_valid = left_cell < num_cells;
            bool limit_left = limiter_enabled && left_valid;
            T dx = faces.centre().x(i_face) - cells.centroids().x(left_cell);
            T dy = faces.centre().y(i_face) - cells.centroids().y(left_cell);
            T dz = faces.centre().z(i_face) - cells.centroids().z(left_cell);

            switch (thermo_interpolator) {
                case ThermoReconstructionVars::rho_p: {
                    T p_limit = limit_left ? limiters.p(left_cell) : 1.0;
                    left.gas.pressure(i_face) =
                        linear_interpolate(flow_states.gas.pressure(left_cell), grad.p,
                                           dx, dy, dz, left_cell, p_limit, left_valid);
                    T rho_limit = (limit_left) ? limiters.rho(left_cell) : 1.0;
                    left.gas.rho(i_face) =
                        linear_interpolate(flow_states.gas.rho(left_cell), grad.rho, dx,
                                           dy, dz, left_cell, rho_limit, left_valid);
                    gas_model.update_thermo_from_rhop(left.gas, i_face);
                    break;
                }
                case ThermoReconstructionVars::rho_u: {
                    T rho_limit = limit_left ? limiters.rho(left_cell) : 1.0;
                    left.gas.rho(i_face) =
                        linear_interpolate(flow_states.gas.rho(left_cell), grad.rho, dx,
                                           dy, dz, left_cell, rho_limit, left_valid);
                    T u_limit = (limit_left) ? limiters.u(left_cell) : 1.0;
                    left.gas.energy(i_face) =
                        linear_interpolate(flow_states.gas.energy(left_cell), grad.u, dx,
                                           dy, dz, left_cell, u_limit, left_valid);
                    gas_model.update_thermo_from_rhou(left.gas, i_face);
                    break;
                }
                case ThermoReconstructionVars::rho_T: {
                    T rho_limit = limit_left ? limiters.rho(left_cell) : 1.0;
                    left.gas.rho(i_face) =
                        linear_interpolate(flow_states.gas.rho(left_cell), grad.rho, dx,
                                           dy, dz, left_cell, rho_limit, left_valid);
                    T T_limit = (limit_left) ? limiters.temp(left_cell) : 1.0;
                    left.gas.temp(i_face) =
                        linear_interpolate(flow_states.gas.temp(left_cell), grad.temp, dx,
                                           dy, dz, left_cell, T_limit, left_valid);
                    gas_model.update_thermo_from_rhoT(left.gas, i_face);
                    break;
                }
                case ThermoReconstructionVars::p_T: {
                    T p_limit = limit_left ? limiters.p(left_cell) : 1.0;
                    left.gas.pressure(i_face) =
                        linear_interpolate(flow_states.gas.pressure(left_cell), grad.p,
                                           dx, dy, dz, left_cell, p_limit, left_valid);
                    T T_limit = (limit_left) ? limiters.temp(left_cell) : 1.0;
                    left.gas.temp(i_face) =
                        linear_interpolate(flow_states.gas.temp(left_cell), grad.temp, dx,
                                           dy, dz, left_cell, T_limit, left_valid);
                    gas_model.update_thermo_from_pT(left.gas, i_face);
                    break;
                }
            }
            T vx_limit = (limit_left) ? limiters.vx(left_cell) : 1.0;
            left.vel.x(i_face) =
                linear_interpolate(flow_states.vel.x(left_cell), grad.vx, dx, dy, dz,
                                   left_cell, vx_limit, left_valid);
            T vy_limit = (limit_left) ? limiters.vy(left_cell) : 1.0;
            left.vel.y(i_face) =
                linear_interpolate(flow_states.vel.y(left_cell), grad.vy, dx, dy, dz,
                                   left_cell, vy_limit, left_valid);
            T vz_limit = (limit_left) ? limiters.vz(left_cell) : 1.0;
            left.vel.z(i_face) =
                linear_interpolate(flow_states.vel.z(left_cell), grad.vz, dx, dy, dz,
                                   left_cell, vz_limit, left_valid);

            // right state
            size_t right_cell = faces.right_cell(i_face);
            bool right_valid = right_cell < num_cells;
            bool limit_right = limiter_enabled && right_valid;

            dx = faces.centre().x(i_face) - cells.centroids().x(right_cell);
            dy = faces.centre().y(i_face) - cells.centroids().y(right_cell);
            dz = faces.centre().z(i_face) - cells.centroids().z(right_cell);

            switch (thermo_interpolator) {
                case ThermoReconstructionVars::rho_p: {
                    T p_limit = (limit_right) ? limiters.p(right_cell) : 1.0;
                    right.gas.pressure(i_face) =
                        linear_interpolate(flow_states.gas.pressure(right_cell), grad.p,
                                           dx, dy, dz, right_cell, p_limit, right_valid);
                    T rho_limit = (limit_right) ? limiters.rho(right_cell) : 1.0;
                    right.gas.rho(i_face) =
                        linear_interpolate(flow_states.gas.rho(right_cell), grad.rho, dx,
                                           dy, dz, right_cell, rho_limit, right_valid);
                    gas_model.update_thermo_from_rhop(right.gas, i_face);
                    break;
                }
                case ThermoReconstructionVars::rho_u: {
                    T u_limit = (limit_right) ? limiters.u(right_cell) : 1.0;
                    right.gas.energy(i_face) =
                        linear_interpolate(flow_states.gas.energy(right_cell), grad.u, dx,
                                           dy, dz, right_cell, u_limit, right_valid);
                    T rho_limit = (limit_right) ? limiters.rho(right_cell) : 1.0;
                    right.gas.rho(i_face) =
                        linear_interpolate(flow_states.gas.rho(right_cell), grad.rho, dx,
                                           dy, dz, right_cell, rho_limit, right_valid);
                    gas_model.update_thermo_from_rhou(right.gas, i_face);
                    break;
                }
                case ThermoReconstructionVars::rho_T: {
                    T T_limit = (limit_right) ? limiters.temp(right_cell) : 1.0;
                    right.gas.temp(i_face) =
                        linear_interpolate(flow_states.gas.temp(right_cell), grad.temp,
                                           dx, dy, dz, right_cell, T_limit, right_valid);
                    T rho_limit = (limit_right) ? limiters.rho(right_cell) : 1.0;
                    right.gas.rho(i_face) =
                        linear_interpolate(flow_states.gas.rho(right_cell), grad.rho, dx,
                                           dy, dz, right_cell, rho_limit, right_valid);
                    gas_model.update_thermo_from_rhoT(right.gas, i_face);
                    break;
                }
                case ThermoReconstructionVars::p_T: {
                    T T_limit = (limit_right) ? limiters.temp(right_cell) : 1.0;
                    right.gas.temp(i_face) =
                        linear_interpolate(flow_states.gas.temp(right_cell), grad.temp,
                                           dx, dy, dz, right_cell, T_limit, right_valid);
                    T p_limit = (limit_right) ? limiters.p(right_cell) : 1.0;
                    right.gas.pressure(i_face) =
                        linear_interpolate(flow_states.gas.pressure(right_cell), grad.p,
                                           dx, dy, dz, right_cell, p_limit, right_valid);
                    gas_model.update_thermo_from_pT(right.gas, i_face);
                    break;
                }
            }
            vx_limit = limit_right ? limiters.vx(right_cell) : 1.0;
            right.vel.x(i_face) =
                linear_interpolate(flow_states.vel.x(right_cell), grad.vx, dx, dy, dz,
                                   right_cell, vx_limit, right_valid);
            vy_limit = (limit_right) ? limiters.vy(right_cell) : 1.0;
            right.vel.y(i_face) =
                linear_interpolate(flow_states.vel.y(right_cell), grad.vy, dx, dy, dz,
                                   right_cell, vy_limit, right_valid);
            vz_limit = (limit_right) ? limiters.vz(right_cell) : 1.0;
            right.vel.z(i_face) =
                linear_interpolate(flow_states.vel.z(right_cell), grad.vz, dx, dy, dz,
                                   right_cell, vz_limit, right_valid);
        });
}

template <typename T>
void ConvectiveFlux<T>::compute_limiters(const FlowStates<T>& flow_states,
                                         const GridBlock<T>& grid,
                                         Gradients<T>& cell_grad) {
    if (limiter_->enabled()) {
        auto cells = grid.cells();
        auto faces = grid.interfaces();
        switch (reconstruction_vars_) {
            case ThermoReconstructionVars::rho_p:
                limiter_->calculate_limiters(flow_states.gas.pressure(), limiters_.p,
                                             cells, faces, cell_grad.p);
                limiter_->calculate_limiters(flow_states.gas.rho(), limiters_.rho, cells,
                                             faces, cell_grad.rho);
                break;
            case ThermoReconstructionVars::rho_u:
                limiter_->calculate_limiters(flow_states.gas.rho(), limiters_.rho, cells,
                                             faces, cell_grad.rho);
                limiter_->calculate_limiters(flow_states.gas.energy(), limiters_.u, cells,
                                             faces, cell_grad.u);
                break;
            case ThermoReconstructionVars::rho_T:
                limiter_->calculate_limiters(flow_states.gas.rho(), limiters_.rho, cells,
                                             faces, cell_grad.rho);
                limiter_->calculate_limiters(flow_states.gas.temp(), limiters_.temp,
                                             cells, faces, cell_grad.temp);
                break;
            case ThermoReconstructionVars::p_T:
                limiter_->calculate_limiters(flow_states.gas.pressure(), limiters_.p,
                                             cells, faces, cell_grad.p);
                limiter_->calculate_limiters(flow_states.gas.temp(), limiters_.temp,
                                             cells, faces, cell_grad.temp);
                break;
        }
        limiter_->calculate_limiters(flow_states.vel.x(), limiters_.vx, cells, faces,
                                     cell_grad.vx);
        limiter_->calculate_limiters(flow_states.vel.y(), limiters_.vy, cells, faces,
                                     cell_grad.vy);
        limiter_->calculate_limiters(flow_states.vel.z(), limiters_.vz, cells, faces,
                                     cell_grad.vz);
    }
}

template class ConvectiveFlux<Ibis::real>;
template class ConvectiveFlux<Ibis::dual>;
