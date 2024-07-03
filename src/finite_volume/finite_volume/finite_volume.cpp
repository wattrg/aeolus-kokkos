#include <finite_volume/boundaries/boundary.h>
#include <finite_volume/conserved_quantities.h>
#include <finite_volume/finite_volume.h>
#include <finite_volume/flux_calc.h>
#include <util/numeric_types.h>

#include <stdexcept>

#include "finite_volume/convective_flux.h"
#include "finite_volume/gradient.h"
#include "gas/transport_properties.h"

template <typename T>
FiniteVolume<T>::FiniteVolume(const GridBlock<T>& grid, json config) {
    dim_ = grid.dim();

    json convective_flux_config = config.at("convective_flux");
    json viscous_flux_config = config.at("viscous_flux");

    // allocate memory for fluxes
    flux_ = ConservedQuantities<T>(grid.num_interfaces(), grid.dim());

    // set up the convective flux
    convective_flux_ = ConvectiveFlux<T>(grid, convective_flux_config);

    // set up the viscous flux
    viscous_flux_ = ViscousFlux<T>(grid, viscous_flux_config);

    // allocate memory for gradients
    size_t reconstruction_order = convective_flux_.reconstruction_order();
    bool viscous = viscous_flux_.enabled();
    if (viscous || reconstruction_order > 1) {
        grad_calc_ = WLSGradient<T>(grid);
        const RequiredGradients grads = convective_flux_.required_gradients();
        cell_grad_ = Gradients<T>(grid.num_cells(), grads.pressure, grads.temp, grads.u,
                                  grads.rho, viscous);
    }

    // set up boundary conditions
    std::vector<std::string> boundary_tags = grid.boundary_tags();
    json boundaries_config = config.at("grid").at("boundaries");
    for (size_t bi = 0; bi < boundary_tags.size(); bi++) {
        // build the actual boundary condition
        json boundary_config = boundaries_config.at(boundary_tags[bi]);
        std::shared_ptr<BoundaryCondition<T>> boundary(
            new BoundaryCondition<T>(boundary_config));
        bcs_.push_back(boundary);

        // the faces associated with this boundary
        bc_interfaces_.push_back(grid.boundary_faces(boundary_tags[bi]));
    }
}

template <typename T>
size_t FiniteVolume<T>::compute_dudt(FlowStates<T>& flow_state, const GridBlock<T>& grid,
                                     ConservedQuantities<T>& dudt, IdealGas<T>& gas_model,
                                     TransportProperties<T>& trans_prop) {
    apply_pre_reconstruction_bc(flow_state, grid, gas_model, trans_prop);
    convective_flux_.compute_convective_flux(flow_state, grid, gas_model, cell_grad_,
                                             grad_calc_, flux_);
    if (viscous_flux_.enabled()) {
        apply_pre_viscous_grad_bc(flow_state, grid, gas_model, trans_prop);
        viscous_flux_.compute_viscous_flux(flow_state, grid, gas_model, trans_prop,
                                           cell_grad_, grad_calc_, flux_);
    }

    flux_surface_integral(grid, dudt);
    return 0;
}

template <typename T>
Ibis::real FiniteVolume<T>::estimate_dt(const FlowStates<T>& flow_state,
                                        GridBlock<T>& grid, IdealGas<T>& gas_model,
                                        TransportProperties<T>& trans_prop) {
    (void)trans_prop;
    size_t num_cells = grid.num_cells();
    CellFaces<T> cell_interfaces = grid.cells().faces();
    Interfaces<T> interfaces = grid.interfaces();
    Cells<T> cells = grid.cells();
    FlowStates<T> face_fs = viscous_flux_.face_fs();
    bool viscous = viscous_flux_.enabled();
    Ibis::real viscous_signal_factor = viscous_flux_.signal_factor();
    // IdealGas<T> gas_model = gas_model_;

    Ibis::real dt;
    Kokkos::parallel_reduce(
        "FV::signal_frequency", num_cells,
        KOKKOS_LAMBDA(const size_t cell_i, Ibis::real& dt_utd) {
            auto cell_face_ids = cell_interfaces.face_ids(cell_i);

            T vx = flow_state.vel.x(cell_i);
            T vy = flow_state.vel.y(cell_i);
            T vz = flow_state.vel.z(cell_i);

            T spectral_radii_c = 0.0;
            T spectral_radii_v = 0.0;
            T volume = cells.volume(cell_i);
            for (size_t face_idx = 0; face_idx < cell_face_ids.size(); face_idx++) {
                size_t i_face = cell_face_ids(face_idx);
                T area = interfaces.area(i_face);
                T dot = vx * interfaces.norm().x(i_face) +
                        vy * interfaces.norm().y(i_face) +
                        vz * interfaces.norm().z(i_face);
                T sig_vel =
                    Ibis::abs(dot) + gas_model.speed_of_sound(flow_state.gas, cell_i);
                spectral_radii_c += sig_vel * area;

                if (viscous) {
                    T gamma = gas_model.gamma();
                    T mu = trans_prop.viscosity(face_fs.gas, gas_model, i_face);
                    T k = trans_prop.thermal_conductivity(face_fs.gas, gas_model, i_face);
                    T rho = face_fs.gas.rho(i_face);
                    T Pr = mu * gas_model.Cp() / k;
                    T tmp = (gamma / rho) * (mu / Pr) * area * area;
                    spectral_radii_v += tmp / volume;
                }
            }
            T local_dt =
                volume / (spectral_radii_c + viscous_signal_factor * spectral_radii_v);
            dt_utd = Ibis::min(Ibis::real_part(local_dt), dt_utd);
        },
        Kokkos::Min<Ibis::real>(dt));

    return dt;
}

template <typename T>
void FiniteVolume<T>::apply_pre_reconstruction_bc(
    FlowStates<T>& fs, const GridBlock<T>& grid, const IdealGas<T>& gas_model,
    const TransportProperties<T>& trans_prop) {
    for (size_t i = 0; i < bcs_.size(); i++) {
        std::shared_ptr<BoundaryCondition<T>> bc = bcs_[i];
        Field<size_t> bc_faces = bc_interfaces_[i];
        bc->apply_pre_reconstruction(fs, grid, bc_faces, gas_model, trans_prop);
    }
}

template <typename T>
void FiniteVolume<T>::apply_pre_viscous_grad_bc(
    FlowStates<T>& fs, const GridBlock<T>& grid, const IdealGas<T>& gas_model,
    const TransportProperties<T>& trans_prop) {
    for (size_t i = 0; i < bcs_.size(); i++) {
        std::shared_ptr<BoundaryCondition<T>> bc = bcs_[i];
        Field<size_t> bc_faces = bc_interfaces_[i];
        bc->apply_pre_viscous_grad(fs, grid, bc_faces, gas_model, trans_prop);
    }
}

template <typename T>
void FiniteVolume<T>::flux_surface_integral(const GridBlock<T>& grid,
                                            ConservedQuantities<T>& dudt) {
    Cells<T> cells = grid.cells();
    CellFaces<T> cell_faces = grid.cells().faces();
    Interfaces<T> faces = grid.interfaces();
    ConservedQuantities<T> flux = flux_;
    size_t num_cells = grid.num_cells();
    Kokkos::parallel_for(
        "flux_integral", num_cells, KOKKOS_LAMBDA(const size_t cell_i) {
            auto face_ids = cell_faces.face_ids(cell_i);
            T d_mass = 0.0;
            T d_momentum_x = 0.0;
            T d_momentum_y = 0.0;
            T d_momentum_z = 0.0;
            T d_energy = 0.0;
            for (size_t face_i = 0; face_i < face_ids.size(); face_i++) {
                size_t face_id = face_ids(face_i);
                T area = -faces.area(face_id) * cell_faces.outsigns(cell_i)(face_i);
                d_mass += flux.mass(face_id) * area;
                d_momentum_x += flux.momentum_x(face_id) * area;
                d_momentum_y += flux.momentum_y(face_id) * area;
                if (flux.dim() == 3) {
                    d_momentum_z += flux.momentum_z(face_id) * area;
                }
                d_energy += flux.energy(face_id) * area;
            }
            dudt.mass(cell_i) = d_mass / cells.volume(cell_i);
            dudt.momentum_x(cell_i) = d_momentum_x / cells.volume(cell_i);
            dudt.momentum_y(cell_i) = d_momentum_y / cells.volume(cell_i);
            if (dudt.dim() == 3) {
                dudt.momentum_z(cell_i) = d_momentum_z / cells.volume(cell_i);
            }
            dudt.energy(cell_i) = d_energy / cells.volume(cell_i);
        });
}

template <typename T>
size_t FiniteVolume<T>::count_bad_cells(const FlowStates<T>& fs, const size_t num_cells) {
    size_t n_bad_cells = 0;
    Kokkos::parallel_reduce(
        "FiniteVolume::count_bad_cells", num_cells,
        KOKKOS_LAMBDA(const int cell_i, size_t& n_bad_cells_utd) {
            if (fs.gas.temp(cell_i) < 0.0 || fs.gas.rho(cell_i) < 0.0 ||
                Ibis::isnan(fs.gas.rho(cell_i)) || Ibis::isinf(fs.gas.rho(cell_i))) {
                n_bad_cells_utd += 1;
            }
        },
        n_bad_cells);
    return n_bad_cells;
}

template <typename T>
void FiniteVolume<T>::compute_viscous_gradient(FlowStates<T>& fs,
                                               const GridBlock<T>& grid,
                                               const IdealGas<T>& gas_model,
                                               const TransportProperties<T>& trans_prop) {
    apply_pre_reconstruction_bc(fs, grid, gas_model, trans_prop);
    apply_pre_viscous_grad_bc(fs, grid, gas_model, trans_prop);
    viscous_flux_.compute_viscous_gradient(fs, grid, cell_grad_, grad_calc_);
}

template <typename T>
void FiniteVolume<T>::compute_convective_gradient(
    FlowStates<T>& fs, const GridBlock<T>& grid, const IdealGas<T>& gas_model,
    const TransportProperties<T>& trans_prop) {
    apply_pre_reconstruction_bc(fs, grid, gas_model, trans_prop);
    convective_flux_.compute_convective_gradient(fs, grid, cell_grad_, grad_calc_);
}

template class FiniteVolume<Ibis::real>;
// template class FiniteVolume<Ibis::dual>;
