#ifndef TRANSPORT_PROPERTIES_H
#define TRANSPORT_PROPERTIES_H

#include <gas/gas_model.h>
#include <gas/gas_state.h>
#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

template <typename T>
class ViscosityModel {
public:
    ViscosityModel() {}

    ViscosityModel(double mu0, double T0, double Ts)
        : mu0_(mu0), T0_(T0), Ts_(Ts) {}

    ViscosityModel(json config) {
        mu0_ = config.at("mu_0");
        T0_ = config.at("T_0");
        Ts_ = config.at("T_s");
    }

    template <typename exec = default_exec_space,
              typename layout = default_layout>
    KOKKOS_INLINE_FUNCTION T viscosity(
        const GasStates<T, layout, typename exec::memory_space>& gas_states,
        const IdealGas<T>& gas_model, const size_t i) const {
        (void)gas_model;
        T temp = gas_states.temp(i);
        return mu0_ * Kokkos::pow(temp, 3 / 2) / (temp + Ts_);
    }

    KOKKOS_INLINE_FUNCTION T viscosity(
        const GasState<T>& gas_state, const IdealGas<T>& gas_model) const {
        (void)gas_model;
        T temp = gas_state.temp;
        return mu0_ * Kokkos::pow(temp, 3 / 2) / (temp + Ts_);
    }

private:
    double mu0_;
    double T0_;
    double Ts_;
};

template <typename T>
class ThermalConductivityModel {
public:
    ThermalConductivityModel() {}

    ThermalConductivityModel(ViscosityModel<T> viscosity, double Pr)
        : viscosity_(viscosity), Pr_(Pr) {}

    ThermalConductivityModel(ViscosityModel<T> viscosity, json config) {
        viscosity_ = viscosity;
        Pr_ = config.at("Pr");
    }

    template <typename exec = default_exec_space,
              typename layout = default_layout>
    KOKKOS_INLINE_FUNCTION T thermal_conductivity(
        const GasStates<T, layout, typename exec::memory_space>& gas_states,
        const IdealGas<T>& gas_model, const size_t i) const {
        T mu = viscosity_.viscosity(gas_states, gas_model, i);
        return gas_model.Cp() * mu / Pr_;
    }

    KOKKOS_INLINE_FUNCTION T thermal_conductivity(
        const GasState<T>& gas_state, const IdealGas<T>& gas_model) const {
        T mu = viscosity_.viscosity(gas_state, gas_model);
        return gas_model.Cp() * mu / Pr_;
    }

private:
    ViscosityModel<T> viscosity_;
    double Pr_;
};

template <typename T>
class TransportProperties {
public:
    TransportProperties() {}

    TransportProperties(ViscosityModel<T> visocity,
                        ThermalConductivityModel<T> thermal_conductivity)
        : viscosity_(visocity), thermal_conductivity_(thermal_conductivity) {}

    TransportProperties(json config) {
        json viscosity_config = config.at("viscosity");
        std::string visocity_type = viscosity_config.at("type");
        if (visocity_type == "sutherland") {
            viscosity_ = ViscosityModel<T>(viscosity_config);
        } else {
            spdlog::error("Unknown viscosity model {}", visocity_type);
            throw new std::runtime_error("Unkown viscosity model");
        }

        json k_config = config.at("thermal_conductivity");
        std::string k_type = k_config.at("type");
        if (k_type == "constant_prandtl_number") {
            thermal_conductivity_ =
                ThermalConductivityModel<T>(viscosity_, k_config);
        } else {
            spdlog::error("Unknown thermal conductivity model {}", k_type);
            throw new std::runtime_error("Unkown thermal conductivity model");
        }
    }

    template <typename exec = default_exec_space,
              typename layout = default_layout>
    KOKKOS_INLINE_FUNCTION T viscosity(
        const GasStates<T, layout, typename exec::memory_space>& gas_states,
        const IdealGas<T>& gas_model, const size_t i) const {
        return viscosity_.viscosity(gas_states, gas_model, i);
    }

    KOKKOS_INLINE_FUNCTION T viscosity(const GasState<T>& gas_state,
                                       const IdealGas<T>& gas_model) const {
        return viscosity_.viscosity(gas_state, gas_model);
    }

    template <typename exec = default_exec_space,
              typename layout = default_layout>
    KOKKOS_INLINE_FUNCTION T thermal_conductivity(
        const GasStates<T, layout, typename exec::memory_space>& gas_states,
        const IdealGas<T>& gas_model, const size_t i) const {
        return thermal_conductivity_.thermal_conductivity(gas_states, gas_model, i);
    }

    KOKKOS_INLINE_FUNCTION T thermal_conductivity(const GasState<T>& gas_state,
                                                  const IdealGas<T>& gas_model) const {
        return thermal_conductivity_.thermal_conductivity(gas_state, gas_model);
    }

private:
    ViscosityModel<T> viscosity_;
    ThermalConductivityModel<T> thermal_conductivity_;
};

#endif
