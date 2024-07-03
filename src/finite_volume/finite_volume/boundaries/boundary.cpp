#include <finite_volume/boundaries/boundary.h>
#include <gas/transport_properties.h>
#include <util/numeric_types.h>
#include <spdlog/spdlog.h>

#include <Kokkos_Core.hpp>


template <typename T>
FlowStateCopy<T>::FlowStateCopy(json flow_state) {
    fs_ = FlowState<T>(flow_state);
}

template <typename T>
void FlowStateCopy<T>::apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                             const Field<size_t>& boundary_faces,
                             const IdealGas<T>& gas_model,
                             const TransportProperties<T>& trans_prop) {
    (void)gas_model;
    (void)trans_prop;

    size_t size = boundary_faces.size();
    auto this_fs = fs_;
    auto interfaces = grid.interfaces();
    size_t num_valid_cells = grid.num_cells();
    Kokkos::parallel_for(
        "FlowStateCopy::apply", size, KOKKOS_LAMBDA(const size_t i) {
            size_t face_id = boundary_faces(i);
            size_t left_cell = interfaces.left_cell(face_id);
            size_t right_cell = interfaces.right_cell(face_id);
            size_t ghost_cell;
            if (left_cell < num_valid_cells) {
                ghost_cell = right_cell;
            } else {
                ghost_cell = left_cell;
            }
            fs.gas.temp(ghost_cell) = this_fs.gas_state.temp;
            fs.gas.pressure(ghost_cell) = this_fs.gas_state.pressure;
            fs.gas.rho(ghost_cell) = this_fs.gas_state.rho;
            fs.gas.energy(ghost_cell) = this_fs.gas_state.energy;

            fs.vel.x(ghost_cell) = this_fs.velocity.x;
            fs.vel.y(ghost_cell) = this_fs.velocity.y;
            fs.vel.z(ghost_cell) = this_fs.velocity.z;
        });
}
template class FlowStateCopy<Ibis::real>;
template class FlowStateCopy<Ibis::dual>;

template <typename T>
BoundaryLayerProfile<T>::BoundaryLayerProfile(json config) {
    std::vector<Ibis::real> x = config.at("height");
    std::vector<Ibis::real> v = config.at("v");
    std::vector<Ibis::real> temp = config.at("T");
    v_ = CubicSpline(x, v);
    T_ = CubicSpline(x, temp);
    p_ = config.at("p");
}

template <typename T>
void BoundaryLayerProfile<T>::apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                                    const Field<size_t>& boundary_faces,
                                    const IdealGas<T>& gas_model,
                                    const TransportProperties<T>& trans_prop) {
    (void)trans_prop;

    size_t size = boundary_faces.size();
    auto interfaces = grid.interfaces();
    auto cells = grid.cells();
    size_t num_valid_cells = grid.num_cells();
    T p = p_;
    CubicSpline temp = T_;
    CubicSpline v = v_;
    Kokkos::parallel_for(
        "FlowStateCopy::apply", size, KOKKOS_LAMBDA(const size_t i) {
            size_t face_id = boundary_faces(i);
            size_t left_cell = interfaces.left_cell(face_id);
            size_t right_cell = interfaces.right_cell(face_id);
            size_t ghost_cell;
            if (left_cell < num_valid_cells) {
                ghost_cell = right_cell;
            } else {
                ghost_cell = left_cell;
            }

            T pos = cells.centroids().y(ghost_cell);
            fs.gas.pressure(ghost_cell) = p;
            fs.gas.temp(ghost_cell) = temp.eval(Ibis::real_part(pos));
            gas_model.update_thermo_from_pT(fs.gas, ghost_cell);

            fs.vel.x(ghost_cell) = v.eval(Ibis::real_part(pos));
            fs.vel.y(ghost_cell) = 0.0;
            fs.vel.z(ghost_cell) = 0.0;
        });
}
template class BoundaryLayerProfile<Ibis::real>;
template class BoundaryLayerProfile<Ibis::dual>;

template <typename T>
void InternalCopy<T>::apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                            const Field<size_t>& boundary_faces,
                            const IdealGas<T>& gas_model,
                            const TransportProperties<T>& trans_prop) {
    (void)gas_model;
    (void)trans_prop;

    size_t size = boundary_faces.size();
    auto interfaces = grid.interfaces();
    size_t num_valid_cells = grid.num_cells();
    Kokkos::parallel_for(
        "InternalCopy::apply", size, KOKKOS_LAMBDA(const size_t i) {
            size_t face_id = boundary_faces(i);

            // determine the valid and the ghost cell
            size_t left_cell = interfaces.left_cell(face_id);
            size_t right_cell = interfaces.right_cell(face_id);
            size_t ghost_cell;
            size_t valid_cell;
            if (left_cell < num_valid_cells) {
                ghost_cell = right_cell;
                valid_cell = left_cell;
            } else {
                ghost_cell = left_cell;
                valid_cell = right_cell;
            }

            // copy data from valid cell to the ghost cell
            fs.gas.temp(ghost_cell) = fs.gas.temp(valid_cell);
            fs.gas.pressure(ghost_cell) = fs.gas.pressure(valid_cell);
            fs.gas.rho(ghost_cell) = fs.gas.rho(valid_cell);
            fs.gas.energy(ghost_cell) = fs.gas.energy(valid_cell);

            fs.vel.x(ghost_cell) = fs.vel.x(valid_cell);
            fs.vel.y(ghost_cell) = fs.vel.y(valid_cell);
            fs.vel.z(ghost_cell) = fs.vel.z(valid_cell);
        });
}
template class InternalCopy<Ibis::real>;
template class InternalCopy<Ibis::dual>;

template <typename T>
void InternalCopyReflectNormal<T>::apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                                         const Field<size_t>& boundary_faces,
                                         const IdealGas<T>& gas_model,
                                         const TransportProperties<T>& trans_prop) {
    (void)gas_model;
    (void)trans_prop;

    size_t size = boundary_faces.size();
    auto interfaces = grid.interfaces();
    size_t num_valid_cells = grid.num_cells();
    Kokkos::parallel_for(
        "ReflectNormal::apply", size, KOKKOS_LAMBDA(const size_t i) {
            size_t face_id = boundary_faces(i);

            // determine the valid and the ghost cell
            size_t left_cell = interfaces.left_cell(face_id);
            size_t right_cell = interfaces.right_cell(face_id);
            size_t ghost_cell;
            size_t valid_cell;
            if (left_cell < num_valid_cells) {
                ghost_cell = right_cell;
                valid_cell = left_cell;
            } else {
                ghost_cell = left_cell;
                valid_cell = right_cell;
            }

            // copy gas state from the valid cell to the ghost cell
            fs.gas.temp(ghost_cell) = fs.gas.temp(valid_cell);
            fs.gas.pressure(ghost_cell) = fs.gas.pressure(valid_cell);
            fs.gas.rho(ghost_cell) = fs.gas.rho(valid_cell);
            fs.gas.energy(ghost_cell) = fs.gas.energy(valid_cell);

            // the velocity in the valid cell
            T x = fs.vel.x(valid_cell);
            T y = fs.vel.y(valid_cell);
            T z = fs.vel.z(valid_cell);

            // the face coordinates
            T norm_x = grid.interfaces().norm().x(face_id);
            T norm_y = grid.interfaces().norm().y(face_id);
            T norm_z = grid.interfaces().norm().z(face_id);
            T tan1_x = grid.interfaces().tan1().x(face_id);
            T tan1_y = grid.interfaces().tan1().y(face_id);
            T tan1_z = grid.interfaces().tan1().z(face_id);
            T tan2_x = grid.interfaces().tan2().x(face_id);
            T tan2_y = grid.interfaces().tan2().y(face_id);
            T tan2_z = grid.interfaces().tan2().z(face_id);

            // the velocity in the valid cell in the interface coordinates
            // with the component normal to the interface negated
            T x_star = -(x * norm_x + y * norm_y + z * norm_z);
            T y_star = x * tan1_x + y * tan1_y + z * tan1_z;
            T z_star = x * tan2_x + y * tan2_y + z * tan2_z;

            // transform the star velocity back to the global frame
            T x_ghost = x_star * norm_x + y_star * tan1_x + z_star * tan2_x;
            T y_ghost = x_star * norm_y + y_star * tan1_y + z_star * tan2_y;
            T z_ghost = x_star * norm_z + y_star * tan1_z + z_star * tan2_z;

            fs.vel.x(ghost_cell) = x_ghost;
            fs.vel.y(ghost_cell) = y_ghost;
            fs.vel.z(ghost_cell) = z_ghost;
        });
}

template <typename T>
void InternalVelCopyReflect<T>::apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                                      const Field<size_t>& boundary_faces,
                                      const IdealGas<T>& gas_model,
                                      const TransportProperties<T>& trans_prop) {
    (void)gas_model;
    (void)trans_prop;

    size_t size = boundary_faces.size();
    auto interfaces = grid.interfaces();
    size_t num_valid_cells = grid.num_cells();
    Kokkos::parallel_for(
        "Reflect::apply", size, KOKKOS_LAMBDA(const size_t i) {
            size_t face_id = boundary_faces(i);

            // determine the valid and the ghost cell
            size_t left_cell = interfaces.left_cell(face_id);
            size_t right_cell = interfaces.right_cell(face_id);
            size_t ghost_cell;
            size_t valid_cell;
            if (left_cell < num_valid_cells) {
                ghost_cell = right_cell;
                valid_cell = left_cell;
            } else {
                ghost_cell = left_cell;
                valid_cell = right_cell;
            }

            // Copy the velocity from the valid cell, but change the sign
            fs.vel.x(ghost_cell) = -fs.vel.x(valid_cell);
            fs.vel.y(ghost_cell) = -fs.vel.y(valid_cell);
            fs.vel.z(ghost_cell) = -fs.vel.z(valid_cell);
        });
}

template <typename T>
void FixTemperature<T>::apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                              const Field<size_t>& boundary_faces,
                              const IdealGas<T>& gas_model,
                              const TransportProperties<T>& trans_prop) {
    (void)gas_model;
    (void)trans_prop;

    size_t size = boundary_faces.size();
    auto interfaces = grid.interfaces();
    size_t num_valid_cells = grid.num_cells();
    T Twall = Twall_;
    Kokkos::parallel_for(
        "FixTemperature", size, KOKKOS_LAMBDA(const size_t i) {
            size_t face_id = boundary_faces(i);

            // determine the valid and the ghost cell
            size_t left_cell = interfaces.left_cell(face_id);
            size_t right_cell = interfaces.right_cell(face_id);
            size_t ghost_cell;
            size_t valid_cell;

            if (left_cell < num_valid_cells) {
                ghost_cell = right_cell;
                valid_cell = left_cell;
            } else {
                ghost_cell = left_cell;
                valid_cell = right_cell;
            }

            // extrapolate the temperature in the ghost cell from the
            // temperature in the valid cell
            fs.gas.temp(ghost_cell) = 2 * Twall - fs.gas.temp(valid_cell);
        });
}

template <typename T>
SubsonicInflow<T>::SubsonicInflow(json flow_state) {
    inflow_state_ = FlowState<T>(flow_state);
}

// the implementation for the subsonic inflow is from Blazek's book
template <typename T>
void SubsonicInflow<T>::apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                              const Field<size_t>& boundary_faces,
                              const IdealGas<T>& gas_model,
                              const TransportProperties<T>& trans_prop) {
    (void)trans_prop;

    size_t size = boundary_faces.size();
    auto interfaces = grid.interfaces();
    size_t num_valid_cells = grid.num_cells();
    FlowState<T> inflow = inflow_state_;
    Kokkos::parallel_for(
        "SubsonicInflow::apply", size, KOKKOS_LAMBDA(const size_t i) {
            // determine the valid and ghost cells
            size_t face_id = boundary_faces(i);
            size_t left_cell = interfaces.left_cell(face_id);
            size_t right_cell = interfaces.right_cell(face_id);
            size_t ghost_cell;
            size_t valid_cell;
            if (left_cell < num_valid_cells) {
                ghost_cell = right_cell;
                valid_cell = left_cell;
            } else {
                ghost_cell = left_cell;
                valid_cell = right_cell;
            }

            // determine the properties at the interface
            T nx = interfaces.norm().x(face_id);
            T ny = interfaces.norm().y(face_id);
            T nz = interfaces.norm().z(face_id);
            T vel_correct = nx * (inflow.velocity.x - fs.vel.x(valid_cell)) +
                            ny * (inflow.velocity.y - fs.vel.y(valid_cell)) +
                            nz * (inflow.velocity.z - fs.vel.z(valid_cell));
            T speed_of_sound = gas_model.speed_of_sound(fs.gas, valid_cell);
            T rho_ref = fs.gas.rho(valid_cell);

            T p_inflow = inflow.gas_state.pressure;
            T p_face = 0.5 * (p_inflow + fs.gas.pressure(valid_cell) -
                              rho_ref * speed_of_sound * vel_correct);
            T rho_face = inflow.gas_state.rho +
                         (p_face - p_inflow) / speed_of_sound / speed_of_sound;
            T vx_face =
                inflow.velocity.x - nx * (p_inflow - p_face) / (rho_ref * speed_of_sound);
            T vy_face =
                inflow.velocity.y - ny * (p_inflow - p_face) / (rho_ref * speed_of_sound);
            T vz_face =
                inflow.velocity.z - nz * (p_inflow - p_face) / (rho_ref * speed_of_sound);

            // extrapolate properties at the interface to the ghost cell
            fs.gas.pressure(ghost_cell) = 2 * p_face - fs.gas.pressure(valid_cell);
            fs.gas.rho(ghost_cell) = 2 * rho_face - fs.gas.rho(valid_cell);
            gas_model.update_thermo_from_rhop(fs.gas, ghost_cell);

            fs.vel.x(ghost_cell) = 2 * vx_face - fs.vel.x(valid_cell);
            fs.vel.y(ghost_cell) = 2 * vy_face - fs.vel.y(valid_cell);
            fs.vel.z(ghost_cell) = 2 * vz_face - fs.vel.z(valid_cell);
        });
}
template class SubsonicInflow<Ibis::real>;
template class SubsonicInflow<Ibis::dual>;

template <typename T>
void SubsonicOutflow<T>::apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                               const Field<size_t>& boundary_faces,
                               const IdealGas<T>& gas_model,
                               const TransportProperties<T>& trans_prop) {
    (void)trans_prop;

    size_t size = boundary_faces.size();
    auto interfaces = grid.interfaces();
    size_t num_valid_cells = grid.num_cells();
    T pressure = pressure_;
    Kokkos::parallel_for(
        "SubsonicOutflow::apply", size, KOKKOS_LAMBDA(const int i) {
            // determine the valid and ghost cells
            size_t face_id = boundary_faces(i);
            size_t left_cell = interfaces.left_cell(face_id);
            size_t right_cell = interfaces.right_cell(face_id);
            size_t ghost_cell;
            size_t valid_cell;
            if (left_cell < num_valid_cells) {
                ghost_cell = right_cell;
                valid_cell = left_cell;
            } else {
                ghost_cell = left_cell;
                valid_cell = right_cell;
            }

            // determine properties at the face
            T nx = interfaces.norm().x(face_id);
            T ny = interfaces.norm().y(face_id);
            T nz = interfaces.norm().z(face_id);
            T p_face = pressure;
            T rho_ref = fs.gas.rho(valid_cell);
            T p_ref = fs.gas.pressure(valid_cell);
            T speed_of_sound = gas_model.speed_of_sound(fs.gas, valid_cell);
            T rho_face = fs.gas.rho(valid_cell) +
                         (p_face - p_ref) / speed_of_sound / speed_of_sound;
            T vx_face =
                fs.vel.x(valid_cell) + nx * (p_ref - p_face) / (rho_ref * speed_of_sound);
            T vy_face =
                fs.vel.y(valid_cell) + ny * (p_ref - p_face) / (rho_ref * speed_of_sound);
            T vz_face =
                fs.vel.z(valid_cell) + nz * (p_ref - p_face) / (rho_ref * speed_of_sound);

            // exptrapolate properties to the ghost cell
            fs.gas.pressure(ghost_cell) = 2 * p_face - fs.gas.pressure(valid_cell);
            fs.gas.rho(ghost_cell) = 2 * rho_face - fs.gas.rho(valid_cell);
            gas_model.update_thermo_from_rhop(fs.gas, ghost_cell);

            fs.vel.x(ghost_cell) = 2 * vx_face - fs.vel.x(valid_cell);
            fs.vel.y(ghost_cell) = 2 * vy_face - fs.vel.y(valid_cell);
            fs.vel.z(ghost_cell) = 2 * vz_face - fs.vel.z(valid_cell);
        });
}
template class SubsonicOutflow<Ibis::real>;
template class SubsonicOutflow<Ibis::dual>;

template <typename T>
std::shared_ptr<BoundaryAction<T>> build_boundary_action(json config) {
    std::string type = config.at("type");
    std::shared_ptr<BoundaryAction<T>> action;
    if (type == "flow_state_copy") {
        json flow_state = config.at("flow_state");
        action = std::shared_ptr<BoundaryAction<T>>(new FlowStateCopy<T>(flow_state));
    } else if (type == "boundary_layer_profile") {
        json profile = config.at("profile");
        action = std::shared_ptr<BoundaryAction<T>>(new BoundaryLayerProfile<T>(profile));
    } else if (type == "internal_copy") {
        action = std::shared_ptr<BoundaryAction<T>>(new InternalCopy<T>());
    } else if (type == "internal_copy_reflect_normal") {
        action = std::shared_ptr<BoundaryAction<T>>(new InternalCopyReflectNormal<T>());
    } else if (type == "internal_vel_copy_reflect") {
        action = std::shared_ptr<BoundaryAction<T>>(new InternalVelCopyReflect<T>());
    } else if (type == "fix_temperature") {
        T temperature = T(config.at("temperature"));
        action = std::shared_ptr<BoundaryAction<T>>(new FixTemperature<T>(temperature));
    } else if (type == "subsonic_inflow") {
        json flow_state = config.at("flow_state");
        action = std::shared_ptr<BoundaryAction<T>>(new SubsonicInflow<T>(flow_state));
    } else if (type == "subsonic_outflow") {
        T pressure = T(config.at("pressure"));
        action = std::shared_ptr<BoundaryAction<T>>(new SubsonicOutflow<T>(pressure));
    } else {
        spdlog::error("Unknown boundary action {}", type);
        throw std::runtime_error("Unknown boundary action");
    }
    return action;
}

template <typename T>
BoundaryCondition<T>::BoundaryCondition(json config) {
    std::vector<json> pre_reco = config.at("pre_reconstruction");
    for (size_t i = 0; i < pre_reco.size(); i++) {
        std::shared_ptr<BoundaryAction<T>> action = build_boundary_action<T>(pre_reco[i]);
        pre_reconstruction_.push_back(action);
    }

    std::vector<json> pre_viscous_grad = config.at("pre_viscous_grad");
    for (size_t i = 0; i < pre_viscous_grad.size(); i++) {
        std::shared_ptr<BoundaryAction<T>> action =
            build_boundary_action<T>(pre_viscous_grad[i]);
        pre_viscous_grad_.push_back(action);
    }
}

template <typename T>
void BoundaryCondition<T>::apply_pre_reconstruction(
    FlowStates<T>& fs, const GridBlock<T>& grid, const Field<size_t>& boundary_faces,
    const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop) {
    for (size_t i = 0; i < pre_reconstruction_.size(); i++) {
        pre_reconstruction_[i]->apply(fs, grid, boundary_faces, gas_model, trans_prop);
    }
}

template <typename T>
void BoundaryCondition<T>::apply_pre_viscous_grad(
    FlowStates<T>& fs, const GridBlock<T>& grid, const Field<size_t>& boundary_faces,
    const IdealGas<T>& gas_model, const TransportProperties<T>& trans_prop) {
    for (size_t i = 0; i < pre_viscous_grad_.size(); i++) {
        pre_viscous_grad_[i]->apply(fs, grid, boundary_faces, gas_model, trans_prop);
    }
}
template class BoundaryCondition<Ibis::real>;
template class BoundaryCondition<Ibis::dual>;
