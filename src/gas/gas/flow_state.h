#ifndef FLOW_STATE_H
#define FLOW_STATE_H

#include <finite_volume/conserved_quantities.h>
#include <gas/gas_state.h>
#include <util/vector3.h>

#include <Kokkos_Core.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

template <typename T>
struct FlowState {
public:
    KOKKOS_INLINE_FUNCTION
    FlowState() {}

    FlowState(json flow_state) {
        gas_state.temp = flow_state.at("T");
        gas_state.pressure = flow_state.at("p");
        gas_state.rho = flow_state.at("rho");
        gas_state.energy = flow_state.at("energy");

        velocity.x = flow_state.at("vx");
        velocity.y = flow_state.at("vy");
        velocity.z = flow_state.at("vz");
    }

    KOKKOS_INLINE_FUNCTION
    FlowState(GasState<T> gs, Vector3<T> vel) : gas_state(gs), velocity(vel) {}

    KOKKOS_INLINE_FUNCTION
    void set_weighted_average(const FlowState<T>& a, T wa, const FlowState<T>& b, T wb) {
        gas_state.rho = wa * a.gas_state.rho + wb * b.gas_state.rho;
        gas_state.pressure = wa * a.gas_state.pressure + wb * b.gas_state.pressure;
        gas_state.temp = wa * a.gas_state.temp + wb * b.gas_state.temp;
        gas_state.energy = wa * a.gas_state.energy + wb * b.gas_state.energy;

        velocity.x = wa * a.velocity.x + wb * b.velocity.x;
        velocity.y = wa * a.velocity.y + wb * b.velocity.y;
        velocity.z = wa * a.velocity.z + wb * b.velocity.z;
    }

    GasState<T> gas_state;
    Vector3<T> velocity;
};

template <typename T, class Layout = Kokkos::DefaultExecutionSpace::array_layout,
          class Space = Kokkos::DefaultExecutionSpace::memory_space>
struct FlowStates {
public:
    using array_layout = Layout;
    using memory_space = Space;
    using host_mirror_mem_space = Kokkos::DefaultHostExecutionSpace::memory_space;
    using host_mirror_layout = Kokkos::DefaultExecutionSpace::array_layout;
    using mirror_type = FlowStates<T, host_mirror_layout, host_mirror_mem_space>;

public:
    FlowStates() {}

    FlowStates(size_t n)
        : gas(GasStates<T, Layout, Space>(n)), vel(Vector3s<T, Layout, Space>(n)) {}

    FlowStates(GasStates<T, Layout, Space> gas, Vector3s<T, Layout, Space> vel)
        : gas(gas), vel(vel) {}

    int number_flow_states() const { return gas.size(); }

    mirror_type host_mirror() const {
        auto gas_mirror = gas.host_mirror();
        auto vel_mirror = vel.host_mirror();
        return mirror_type(gas_mirror, vel_mirror);
    }

    template <class OtherSpace>
    void deep_copy(const FlowStates<T, Layout, OtherSpace>& other) {
        gas.deep_copy(other.gas);
        vel.deep_copy(other.vel);
    }

    KOKKOS_INLINE_FUNCTION
    void set_flow_state(const FlowState<T>& other, const size_t i) const {
        gas.set_gas_state(other.gas_state, i);
        vel.set_vector(other.velocity, i);
    }

    KOKKOS_INLINE_FUNCTION
    FlowState<T> average_flow_states_pT(const size_t a, const size_t b) const {
        return FlowState<T>{gas.average_pT(a, b), vel.average_vectors(a, b)};
    }

    KOKKOS_INLINE_FUNCTION
    FlowState<T> flow_state(size_t i) const {
        GasState<T> gs;
        gs.temp = gas.temp(i);
        gs.pressure = gas.pressure(i);
        gs.energy = gas.energy(i);
        gs.rho = gas.rho(i);

        Vector3<T> velocity;
        velocity.x = vel.x(i);
        velocity.y = vel.y(i);
        velocity.z = vel.z(i);
        return FlowState<T>{gs, velocity};
    }

    GasStates<T, Layout, Space> gas;
    Vector3s<T, Layout, Space> vel;
};

#endif
