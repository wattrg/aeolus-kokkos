#include <finite_volume/boundaries/boundary.h>
#include <spdlog/spdlog.h>

#include <Kokkos_Core.hpp>
#include "gas/transport_properties.h"

template <typename T>
FlowStateCopy<T>::FlowStateCopy(json flow_state) {
    T temp = flow_state.at("T");
    T pressure = flow_state.at("p");
    T rho = flow_state.at("rho");
    T energy = flow_state.at("energy");

    GasState<T> gs;
    gs.temp = temp;
    gs.pressure = pressure;
    gs.rho = rho;
    gs.energy = energy;

    T vx = flow_state.at("vx");
    T vy = flow_state.at("vy");
    T vz = flow_state.at("vz");
    Vector3<T> vel{vx, vy, vz};
    fs_ = FlowState<T>(gs, vel);
}

template <typename T>
void FlowStateCopy<T>::apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                             const Field<size_t>& boundary_faces,
                             const IdealGas<T>& gas_model,
                             const TransportProperties<T>& trans_prop) {
    (void) gas_model;
    (void) trans_prop;

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
template class FlowStateCopy<double>;

template <typename T>
BoundaryLayerProfile<T>::BoundaryLayerProfile(json config) {
    std::vector<double> x = config.at("height");
    std::vector<double> v = config.at("v");
    std::vector<double> temp = config.at("T");
    v_ = CubicSpline(x, v);
    T_ = CubicSpline(x, temp);
    p_ = config.at("p");
}

template <typename T>
void BoundaryLayerProfile<T>::apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                        const Field<size_t>& boundary_faces,
                        const IdealGas<T>& gas_model,
                        const TransportProperties<T>& trans_prop) {
    (void) trans_prop;

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
            }
            else {
                ghost_cell = left_cell;
            }
            T pos = cells.centroids().y(ghost_cell);
            fs.gas.pressure(ghost_cell) = p;
            fs.gas.temp(ghost_cell) = temp.eval(pos);
            gas_model.update_thermo_from_pT(fs.gas, ghost_cell);

            fs.vel.x(ghost_cell) = v.eval(pos);
            fs.vel.y(ghost_cell) = 0.0;
            fs.vel.z(ghost_cell) = 0.0;
        }  
    );
}
template class BoundaryLayerProfile<double>;

template <typename T>
void InternalCopy<T>::apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                            const Field<size_t>& boundary_faces,
                            const IdealGas<T>& gas_model,
                            const TransportProperties<T>& trans_prop) {
    (void) gas_model;
    (void) trans_prop;
    
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
template class InternalCopy<double>;

template <typename T>
void InternalCopyReflectNormal<T>::apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                                         const Field<size_t>& boundary_faces,
                                         const IdealGas<T>& gas_model,
                                         const TransportProperties<T>& trans_prop) {
    (void) gas_model;
    (void) trans_prop;

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
    (void) gas_model;
    (void) trans_prop;

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
    (void) gas_model;
    (void) trans_prop;

    size_t size = boundary_faces.size();
    auto interfaces = grid.interfaces();
    size_t num_valid_cells = grid.num_cells();
    double Twall = Twall_;
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
        double temperature = config.at("temperature");
        action = std::shared_ptr<BoundaryAction<T>>(new FixTemperature<T>(temperature));
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
void BoundaryCondition<T>::apply_pre_reconstruction(FlowStates<T>& fs,
                                                    const GridBlock<T>& grid,
                                                    const Field<size_t>& boundary_faces,
                                                    const IdealGas<T>& gas_model,
                                                    const TransportProperties<T>& trans_prop) {
    for (size_t i = 0; i < pre_reconstruction_.size(); i++) {
        pre_reconstruction_[i]->apply(fs, grid, boundary_faces, gas_model, trans_prop);
    }
}

template <typename T>
void BoundaryCondition<T>::apply_pre_viscous_grad(FlowStates<T>& fs,
                                                  const GridBlock<T>& grid,
                                                  const Field<size_t>& boundary_faces,
                                                  const IdealGas<T>& gas_model,
                                                  const TransportProperties<T>& trans_prop) {
    for (size_t i = 0; i < pre_viscous_grad_.size(); i++) {
        pre_viscous_grad_[i]->apply(fs, grid, boundary_faces, gas_model, trans_prop);
    }
}
template class BoundaryCondition<double>;
