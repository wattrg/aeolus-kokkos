#include <finite_volume/conserved_quantities.h>
#include <util/numeric_types.h>

template <typename T>
void ConservedQuantitiesNorm<T>::write_to_file(std::ofstream& f, Ibis::real time,
                                               size_t step) {
    f << time << " " << step << " " << Ibis::real_part(mass()) << " "
      << Ibis::real_part(momentum_x()) << " " << Ibis::real_part(momentum_y()) << " "
      << Ibis::real_part(momentum_z()) << " " << Ibis::real_part(energy()) << std::endl;
}
template class ConservedQuantitiesNorm<Ibis::real>;
template class ConservedQuantitiesNorm<Ibis::dual>;

template <typename T>
ConservedQuantities<T>::ConservedQuantities(size_t n, size_t dim)
    : cq_(Kokkos::View<T**>("ConservedQuantities", n, dim + 2)),
      num_values_(n),
      dim_(dim) {
    mass_idx_ = 0;
    momentum_idx_ = 1;
    energy_idx_ = momentum_idx_ + dim;
}

template <typename T>
void ConservedQuantities<T>::apply_time_derivative(const ConservedQuantities<T>& dudt,
                                                   Ibis::real dt) {
    Kokkos::parallel_for(
        "CQ::update_cq", num_values_, KOKKOS_CLASS_LAMBDA(const size_t i) {
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
ConservedQuantitiesNorm<T> ConservedQuantities<T>::L2_norms() const {
    ConservedQuantitiesNorm<T> norms{};
    Kokkos::parallel_reduce(
        "L2_norm", num_values_,
        KOKKOS_CLASS_LAMBDA(const size_t i, ConservedQuantitiesNorm<T>& tl_cq) {
            tl_cq.mass() += mass(i) * mass(i);
            tl_cq.momentum_x() += momentum_x(i) * momentum_x(i);
            tl_cq.momentum_y() += momentum_y(i) * momentum_y(i);
            if (dim_ == 3) {
                tl_cq.momentum_z() += momentum_z(i) * momentum_z(i);
            }
            tl_cq.energy() += energy(i) * energy(i);
        },
        Kokkos::Sum<ConservedQuantitiesNorm<T>>(norms));

    norms.mass() = Ibis::sqrt(norms.mass());
    norms.momentum_x() = Ibis::sqrt(norms.momentum_x());
    norms.momentum_y() = Ibis::sqrt(norms.momentum_y());
    norms.momentum_z() = Ibis::sqrt(norms.momentum_z());
    norms.energy() = Ibis::sqrt(norms.energy());
    return norms;
}

template <typename T>
void ConservedQuantities<T>::deep_copy(const ConservedQuantities<T>& other) {
    Kokkos::deep_copy(cq_, other.cq_);
}

template class ConservedQuantities<Ibis::real>;
template class ConservedQuantities<Ibis::dual>;

template <typename T>
void apply_time_derivative(const ConservedQuantities<T>& U0, ConservedQuantities<T>& U1,
                           ConservedQuantities<T>& dUdt, Ibis::real dt) {
    size_t n_values = U0.size();
    size_t dim = U0.dim();
    Kokkos::parallel_for(
        "apply_time_derivative", n_values, KOKKOS_LAMBDA(const size_t i) {
            U1.mass(i) = U0.mass(i) + dUdt.mass(i) * dt;
            U1.momentum_x(i) = U0.momentum_x(i) + dUdt.momentum_x(i) * dt;
            U1.momentum_y(i) = U0.momentum_y(i) + dUdt.momentum_y(i) * dt;
            if (dim == 3) {
                U1.momentum_z(i) = U0.momentum_z(i) + dUdt.momentum_z(i) * dt;
            }
            U1.energy(i) = U0.energy(i) + dUdt.energy(i) * dt;
        });
}

template void apply_time_derivative(const ConservedQuantities<Ibis::real>&,
                                    ConservedQuantities<Ibis::real>&,
                                    ConservedQuantities<Ibis::real>&, Ibis::real);
template void apply_time_derivative(const ConservedQuantities<Ibis::dual>&,
                                    ConservedQuantities<Ibis::dual>&,
                                    ConservedQuantities<Ibis::dual>&, Ibis::real);
