#ifndef GAS_MODEL_H
#define GAS_MODEL_H

#include <gas/gas_state.h>
#include <util/numeric_types.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

template <typename T>
KOKKOS_INLINE_FUNCTION T rho_from_pT(T p, T temp, T R) {
    return p / (R * temp);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T T_from_rhop(T rho, T p, T R) {
    return p / (rho * R);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T p_from_rhoT(T rho, T temp, T R) {
    return rho * R * temp;
}

template <typename T>
KOKKOS_INLINE_FUNCTION T speed_of_sound_(T temp, T R, T gamma) {
    // assert(Ibis::real_part(temp) > 0.0);
    return Ibis::sqrt(gamma * R * temp);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T temp_from_energy(T u, T Cv) {
    return u / Cv;
}

template <typename T>
KOKKOS_INLINE_FUNCTION T energy_from_temp(T temp, T Cv) {
    return Cv * temp;
}

using default_layout = Kokkos::DefaultExecutionSpace::array_layout;
using default_space = Kokkos::DefaultExecutionSpace::memory_space;
using default_exec_space = Kokkos::DefaultExecutionSpace;

template <typename T>
class IdealGas {
public:
    IdealGas() {}

    IdealGas(Ibis::real R) {
        R_ = T(R);
        Cv_ = 5.0 / 2.0 * R;
        Cp_ = 7.0 / 2.0 * R;
        gamma_ = Cp_ / Cv_;
    }

    IdealGas(json config)
        : R_(config.at("R")),
          Cv_(config.at("Cv")),
          Cp_(config.at("Cp")),
          gamma_(config.at("gamma")) {}

    // update an individual gas state
    KOKKOS_INLINE_FUNCTION
    void update_thermo_from_pT(GasState<T> &gs) const {
        gs.rho = rho_from_pT(gs.pressure, gs.temp, R_);
        gs.energy = energy_from_temp(gs.temp, Cv_);
    }

    KOKKOS_INLINE_FUNCTION
    void update_thermo_from_rhoT(GasState<T> &gs) const {
        gs.pressure = p_from_rhoT(gs.rho, gs.temp, R_);
        gs.energy = energy_from_temp(gs.temp, Cv_);
    }

    KOKKOS_INLINE_FUNCTION
    void update_thermo_from_rhop(GasState<T> &gs) const {
        gs.temp = T_from_rhop(gs.rho, gs.pressure, R_);
        gs.energy = energy_from_temp(gs.temp, Cv_);
    }

    KOKKOS_INLINE_FUNCTION
    void update_thermo_from_rhou(GasState<T> &gs) const {
        gs.temp = temp_from_energy(gs.energy, Cv_);
        gs.pressure = p_from_rhoT(gs.rho, gs.temp, R_);
    }

    // update a single gas state from the collection
    template <typename layout = default_layout, typename space = default_space>
    KOKKOS_INLINE_FUNCTION void update_thermo_from_pT(
        const GasStates<T, layout, space> &gs, const size_t i) const {
        gs.rho(i) = rho_from_pT(gs.pressure(i), gs.temp(i), R_);
        gs.energy(i) = energy_from_temp(gs.temp(i), Cv_);
    }

    template <typename layout = default_layout, typename space = default_space>
    KOKKOS_INLINE_FUNCTION void update_thermo_from_rhoT(
        const GasStates<T, layout, space> &gs, const size_t i) const {
        gs.pressure(i) = p_from_rhoT(gs.rho(i), gs.temp(i), R_);
        gs.energy(i) = energy_from_temp(gs.temp(i), Cv_);
    }

    template <typename layout = default_layout, typename space = default_space>
    KOKKOS_INLINE_FUNCTION void update_thermo_from_rhop(
        const GasStates<T, layout, space> &gs, const size_t i) const {
        gs.temp(i) = T_from_rhop(gs.rho(i), gs.pressure(i), R_);
        gs.energy(i) = energy_from_temp(gs.temp(i), Cv_);
    }

    template <typename layout = default_layout, typename space = default_space>
    KOKKOS_INLINE_FUNCTION void update_thermo_from_rhou(
        const GasStates<T, layout, space> &gs, const size_t i) const {
        gs.temp(i) = temp_from_energy(gs.energy(i), Cv_);
        gs.pressure(i) = p_from_rhoT(gs.rho(i), gs.temp(i), R_);
    }

    // update all the gas states
    template <typename exec = default_exec_space, typename layout = default_layout>
    void update_thermo_from_pT(
        GasStates<T, layout, typename exec::memory_space> &gs) const {
        Kokkos::parallel_for(
            "update_thermo_from_pT", Kokkos::RangePolicy<exec>(0, gs.size()),
            KOKKOS_CLASS_LAMBDA(const size_t i) { update_thermo_from_pT(gs, i); });
    }

    template <typename exec = default_exec_space, typename layout = default_layout>
    void update_thermo_from_rhoT(
        GasStates<T, layout, typename exec::memory_space> &gs) const {
        Kokkos::parallel_for(
            "update_thermo_from_rhoT", Kokkos::RangePolicy<exec>(0, gs.size()),
            KOKKOS_CLASS_LAMBDA(const size_t i) { update_thermo_from_rhoT(gs, i); });
    }

    template <typename exec = default_exec_space, typename layout = default_layout>
    void update_thermo_from_rhop(
        GasStates<T, layout, typename exec::memory_space> &gs) const {
        Kokkos::parallel_for(
            "update_thermo_from_rhop", Kokkos::RangePolicy<exec>(0, gs.size()),
            KOKKOS_CLASS_LAMBDA(const size_t i) { update_thermo_from_rhop(gs, i); });
    }

    template <typename exec = default_exec_space, typename layout = default_layout>
    void update_thermo_from_rhou(
        GasStates<T, layout, typename exec::memory_space> &gs) const {
        Kokkos::parallel_for(
            "update_thermo_from_rhou", Kokkos::RangePolicy<exec>(0, gs.size()),
            KOKKOS_CLASS_LAMBDA(const size_t i) { update_thermo_from_rhou(gs, i); });
    }

    // speed of sound
    T speed_of_sound(const GasState<T> &gs) const {
        return speed_of_sound_(gs.temp, R_, gamma_);
    }

    template <typename layout = default_layout, typename space = default_space>
    KOKKOS_INLINE_FUNCTION T speed_of_sound(const GasStates<T, layout, space> &gs,
                                            const size_t i) const {
        return speed_of_sound_(gs.temp(i), R_, gamma_);
    }

    T internal_energy(const GasState<T> &gs) const { return Cv_ * gs.temp; }

    template <typename layout = default_layout, typename space = default_space>
    KOKKOS_INLINE_FUNCTION T internal_energy(const GasStates<T, layout, space> &gs,
                                             const size_t i) const {
        return Cv_ * gs.temp(i);
    }

    KOKKOS_INLINE_FUNCTION
    T R() const { return R_; }

    KOKKOS_INLINE_FUNCTION
    T Cv() const { return Cv_; }

    KOKKOS_INLINE_FUNCTION
    T Cp() const { return Cp_; }

    KOKKOS_INLINE_FUNCTION
    T gamma() const { return gamma_; }

private:
    T R_;
    T Cv_;
    T Cp_;
    T gamma_;
};

#endif
