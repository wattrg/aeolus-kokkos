#include <finite_volume/viscous_flux.h>

#include "finite_volume/conserved_quantities.h"
#include "finite_volume/gradient.h"
#include "gas/transport_properties.h"

template <typename T>
struct ViscousProperties {
    FlowState<T> flow;
    Vector3<T> grad_temp;
    Vector3<T> grad_vx;
    Vector3<T> grad_vy;
    Vector3<T> grad_vz;
};

template <typename T>
KOKKOS_INLINE_FUNCTION void copy_gradients_to_face(ViscousProperties<T>& props,
                                                   const Gradients<T>& cell_grad,
                                                   const size_t interior_cell) {
    props.grad_temp.x = cell_grad.temp.x(interior_cell);
    props.grad_temp.y = cell_grad.temp.y(interior_cell);
    props.grad_temp.z = cell_grad.temp.z(interior_cell);

    props.grad_vx.x = cell_grad.vx.x(interior_cell);
    props.grad_vx.y = cell_grad.vx.y(interior_cell);
    props.grad_vx.z = cell_grad.vx.z(interior_cell);

    props.grad_vy.x = cell_grad.vy.x(interior_cell);
    props.grad_vy.y = cell_grad.vy.y(interior_cell);
    props.grad_vy.z = cell_grad.vy.z(interior_cell);

    props.grad_vz.x = cell_grad.vz.x(interior_cell);
    props.grad_vz.y = cell_grad.vz.y(interior_cell);
    props.grad_vz.z = cell_grad.vz.z(interior_cell);
}

template <typename T>
KOKKOS_INLINE_FUNCTION Vector3<T> hasselbacher_average_(
    const Vector3s<T>& grad, const T value_left, const T value_right, const size_t left,
    const size_t right, const Vector3<T>& ehat, const Vector3<T>& n, const T& len_e,
    const T& ehat_dot_n) {
    T avg_grad_x = 0.5 * (grad.x(left) + grad.x(right));
    T avg_grad_y = 0.5 * (grad.y(left) + grad.y(right));
    T avg_grad_z = 0.5 * (grad.z(left) + grad.z(right));
    T avg_dot_ehat = avg_grad_x * ehat.x + avg_grad_y * ehat.y + avg_grad_z * ehat.z;
    T correction = avg_dot_ehat - (value_right - value_left) / len_e;

    return Vector3<T>{avg_grad_x - correction * n.x / ehat_dot_n,
                      avg_grad_y - correction * n.y / ehat_dot_n,
                      avg_grad_z - correction * n.z / ehat_dot_n};
}

template <typename T>
KOKKOS_INLINE_FUNCTION void hasselbacher_average(
    ViscousProperties<T>& props, const Gradients<T>& cell_grad, const Cells<T>& cells,
    const FlowStates<T>& fs, const Interfaces<T>& faces, const size_t left_cell,
    const size_t right_cell, const size_t face) {
    // Vector from right cell centre to left cell centre
    T ex = cells.centroids().x(right_cell) - cells.centroids().x(left_cell);
    T ey = cells.centroids().y(right_cell) - cells.centroids().y(left_cell);
    T ez = cells.centroids().z(right_cell) - cells.centroids().z(left_cell);

    // Some properties of the grid used by Hasselbacher averaging
    T len_e = Ibis::sqrt(ex * ex + ey * ey + ez * ez);
    Vector3<T> ehat{ex / len_e, ey / len_e, ez / len_e};
    Vector3<T> n{faces.norm().x(face), faces.norm().y(face), faces.norm().z(face)};
    T ehat_dot_n = ehat.x * n.x + ehat.y * n.y + ehat.z * n.z;

    props.grad_vx =
        hasselbacher_average_(cell_grad.vx, fs.vel.x(left_cell), fs.vel.x(right_cell),
                              left_cell, right_cell, ehat, n, len_e, ehat_dot_n);
    props.grad_vy =
        hasselbacher_average_(cell_grad.vy, fs.vel.y(left_cell), fs.vel.y(right_cell),
                              left_cell, right_cell, ehat, n, len_e, ehat_dot_n);
    props.grad_vz =
        hasselbacher_average_(cell_grad.vz, fs.vel.z(left_cell), fs.vel.z(right_cell),
                              left_cell, right_cell, ehat, n, len_e, ehat_dot_n);
    props.grad_temp = hasselbacher_average_(cell_grad.temp, fs.gas.temp(left_cell),
                                            fs.gas.temp(right_cell), left_cell,
                                            right_cell, ehat, n, len_e, ehat_dot_n);
}

template <typename T>
KOKKOS_FUNCTION ViscousProperties<T> compute_viscous_properties_at_faces(
    const FlowStates<T>& flow_states, const Interfaces<T>& faces, const Cells<T>& cells,
    const IdealGas<T>& gas_model, const Gradients<T>& cell_grad, const size_t num_cells,
    const size_t face_i) {
    ViscousProperties<T> props;
    size_t left_cell = faces.left_cell(face_i);
    size_t right_cell = faces.right_cell(face_i);
    bool left_valid = left_cell < num_cells;
    bool right_valid = right_cell < num_cells;

    // get the viscous gradients at faces
    if (!left_valid || !right_valid) {
        size_t interior_cell = (left_valid) ? left_cell : right_cell;
        copy_gradients_to_face(props, cell_grad, interior_cell);
    } else {
        hasselbacher_average(props, cell_grad, cells, flow_states, faces, left_cell,
                             right_cell, face_i);
    }

    // get the flow state at faces
    props.flow = flow_states.average_flow_states_pT(left_cell, right_cell);
    gas_model.update_thermo_from_pT(props.flow.gas_state);

    return props;
}

template <typename T>
ViscousFlux<T>::ViscousFlux(const GridBlock<T>& grid, json config) {
    enabled_ = config.at("enabled");
    signal_factor_ = config.at("signal_factor");
    (void)grid;

    if (enabled_) {
        face_fs_ = FlowStates<T>(grid.num_interfaces());
    }
}

template <typename T>
void ViscousFlux<T>::compute_viscous_gradient(const FlowStates<T>& flow_states,
                                              const GridBlock<T>& grid,
                                              Gradients<T>& cell_grad,
                                              WLSGradient<T>& grad_calc) {
    grad_calc.compute_gradients(grid, flow_states.gas.temp(), cell_grad.temp);
    grad_calc.compute_gradients(grid, flow_states.vel.x(), cell_grad.vx);
    grad_calc.compute_gradients(grid, flow_states.vel.y(), cell_grad.vy);
    grad_calc.compute_gradients(grid, flow_states.vel.z(), cell_grad.vz);
}

template <typename T>
void ViscousFlux<T>::compute_viscous_flux(
    const FlowStates<T>& flow_states, const GridBlock<T>& grid,
    const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop,
    Gradients<T>& cell_grad, WLSGradient<T>& grad_calc, ConservedQuantities<T>& flux) {
    compute_viscous_gradient(flow_states, grid, cell_grad, grad_calc);

    size_t num_faces = grid.num_interfaces();
    Interfaces<T> interfaces = grid.interfaces();
    Cells<T> cells = grid.cells();
    const size_t num_cells = grid.num_cells();
    FlowStates<T> face_fs = face_fs_;
    // Gradients<T> grad = face_grad_;
    size_t dim = grid.dim();
    Kokkos::parallel_for(
        "viscous_flux", num_faces, KOKKOS_LAMBDA(const size_t i) {
            auto props = compute_viscous_properties_at_faces(
                flow_states, interfaces, cells, gas_model, cell_grad, num_cells, i);

            face_fs.set_flow_state(props.flow, i);

            // transport properties at the face
            T mu = trans_prop.viscosity(props.flow.gas_state, gas_model);
            T k = trans_prop.thermal_conductivity(props.flow.gas_state, gas_model);
            T lambda = -2.0 / 3.0 * mu;

            // compute the viscous fluxes
            T bulk = lambda * (props.grad_vx.x + props.grad_vy.y + props.grad_vz.z);
            T tau_xx = 2.0 * mu * props.grad_vx.x + bulk;
            T tau_yy = 2.0 * mu * props.grad_vy.y + bulk;
            T tau_zz = 2.0 * mu * props.grad_vz.z + bulk;
            T tau_xy = mu * (props.grad_vx.y + props.grad_vy.x);
            T tau_xz = mu * (props.grad_vx.z + props.grad_vz.x);
            T tau_yz = mu * (props.grad_vy.z + props.grad_vz.y);

            T vx = props.flow.velocity.x;
            T vy = props.flow.velocity.y;
            T vz = props.flow.velocity.z;
            T theta_x = vx * tau_xx + vy * tau_xy + vz * tau_xz + k * props.grad_temp.x;
            T theta_y = vx * tau_xy + vy * tau_yy + vz * tau_yz + k * props.grad_temp.y;
            T theta_z = vx * tau_xz + vy * tau_yz + vz * tau_zz + k * props.grad_temp.z;

            T nx = interfaces.norm().x(i);
            T ny = interfaces.norm().y(i);
            T nz = interfaces.norm().z(i);
            flux.momentum_x(i) -= nx * tau_xx + ny * tau_xy + nz * tau_xz;
            flux.momentum_y(i) -= nx * tau_xy + ny * tau_yy + nz * tau_yz;
            if (dim == 3) {
                flux.momentum_z(i) -= nx * tau_xz + ny * tau_yz + nz * tau_zz;
            }
            flux.energy(i) -= nx * theta_x + ny * theta_y + nz * theta_z;
        });
}

template class ViscousFlux<Ibis::real>;
template class ViscousFlux<Ibis::dual>;
