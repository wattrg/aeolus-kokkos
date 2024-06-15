#include <finite_volume/conserved_quantities.h>

template <typename T>
void ConservedQuantitiesNorm<T>::write_to_file(std::ofstream& f, double time, unsigned int step) {
    f << time << " " << step << " " << mass() << " " << momentum_x() << " " << momentum_y() << " " << momentum_z() << " " << energy() << std::endl;
}
template class ConservedQuantitiesNorm<double>;

template <typename T>
ConservedQuantities<T>::ConservedQuantities(unsigned int n, unsigned int dim)
    : cq_(Kokkos::View<T**>("ConservedQuantities", n, dim + 2)),
      num_values_(n),
      dim_(dim) {
    mass_idx_ = 0;
    momentum_idx_ = 1;
    energy_idx_ = momentum_idx_ + dim;
}

template <typename T>
void ConservedQuantities<T>::apply_time_derivative(const ConservedQuantities<T>& dudt,
                                                   double dt) {
    Kokkos::parallel_for(
        "CQ::update_cq", num_values_, KOKKOS_CLASS_LAMBDA(const int i) {
            mass(i) += dudt.mass(i) * dt;
            momentum_x(i) += dudt.momentum_x(i) * dt;
            momentum_y(i) += dudt.momentum_y(i) * dt;
            if (dim_ == 3) {
                momentum_z(i) += dudt.momentum_z(i) * dt;
            }
            energy(i) += dudt.energy(i) * dt;
        });
}

template <typename T>
ConservedQuantitiesNorm<double> ConservedQuantities<T>::L2_norms() const {
    ConservedQuantitiesNorm<double> norms{};
    Kokkos::parallel_reduce(
        "L2_norm", num_values_,
        KOKKOS_CLASS_LAMBDA(const int i, ConservedQuantitiesNorm<double>& tl_cq) {
            tl_cq.mass() += mass(i) * mass(i);
            tl_cq.momentum_x() += momentum_x(i) * momentum_x(i);
            tl_cq.momentum_y() += momentum_y(i) * momentum_y(i);
            if (dim_ == 3) {
                tl_cq.momentum_z() += momentum_z(i) * momentum_z(i);
            }
            tl_cq.energy() += energy(i) * energy(i);
        },
        Kokkos::Sum<ConservedQuantitiesNorm<double>>(norms));

    norms.mass() = Kokkos::sqrt(norms.mass());
    norms.momentum_x() = Kokkos::sqrt(norms.momentum_x());
    norms.momentum_y() = Kokkos::sqrt(norms.momentum_y());
    norms.momentum_z() = Kokkos::sqrt(norms.momentum_z());
    norms.energy() = Kokkos::sqrt(norms.energy());
    return norms;
}

template class ConservedQuantities<double>;

template <typename T>
void apply_time_derivative(const ConservedQuantities<T>& U0, ConservedQuantities<T>& U1,
                           ConservedQuantities<T>& dUdt, double dt) {
    size_t n_values = U0.size();
    size_t dim = U0.dim();
    Kokkos::parallel_for(
        "apply_time_derivative", n_values, KOKKOS_LAMBDA(const int i) {
            U1.mass(i) = U0.mass(i) + dUdt.mass(i) * dt;
            U1.momentum_x(i) = U0.momentum_x(i) + dUdt.momentum_x(i) * dt;
            U1.momentum_y(i) = U0.momentum_y(i) + dUdt.momentum_y(i) * dt;
            if (dim == 3) {
                U1.momentum_z(i) = U0.momentum_z(i) + dUdt.momentum_z(i) * dt;
            }
            U1.energy(i) = U0.energy(i) + dUdt.energy(i) * dt;
        });
}

template void apply_time_derivative(const ConservedQuantities<double>&,
                                    ConservedQuantities<double>&,
                                    ConservedQuantities<double>&, double);
