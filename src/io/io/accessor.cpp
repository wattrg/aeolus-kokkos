#include <io/accessor.h>
#include "finite_volume/gradient.h"

template <typename T>
T PressureAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs,
    const IdealGas<T>& gas_model, const int i) {
    (void) gas_model;
    return fs.gas.pressure(i);
}
template class PressureAccess<double>;

template <typename T>
T TemperatureAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs,
    const IdealGas<T>& gas_model, const int i) {
    (void) gas_model;
    return fs.gas.temp(i);
}
template class TemperatureAccess<double>;

template <typename T>
T DensityAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs,
    const IdealGas<T>& gas_model, const int i) {
    (void) gas_model;
    return fs.gas.rho(i);
}
template class DensityAccess<double>;

template <typename T>
T InternalEnergyAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs,
    const IdealGas<T>& gas_model, const int i) {
    (void)gas_model;
    return fs.gas.energy(i);
}
template class InternalEnergyAccess<double>;

template <typename T>
T SpeedOfSoundAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs,
    const IdealGas<T>& gas_model, const int i) {
    return gas_model.speed_of_sound(fs.gas, i);
}
template class SpeedOfSoundAccess<double>;

template <typename T>
T MachNumberAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs,
    const IdealGas<T>& gas_model, const int i) {
    T a = gas_model.speed_of_sound(fs.gas, i);
    T vx = fs.vel.x(i);
    T vy = fs.vel.y(i);
    T vz = fs.vel.z(i);
    T v_mag = Kokkos::sqrt(vx * vx + vy * vy + vz * vz);
    return v_mag / a;
}
template class MachNumberAccess<double>;

template <typename T>
Vector3<T> VelocityAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs,
    const IdealGas<T>& gas_model, const int i) {
    
    (void) gas_model;
    T x = fs.vel.x(i);
    T y = fs.vel.x(i);
    T z = fs.vel.x(i);
    return Vector3<T>(x, y, z);
}
template class VelocityAccess<double>;

template <typename T>
void GradVxAccess<T>::init(const FlowStates<T, array_layout, host_mem_space>& fs,
                           const typename GridBlock<T>::mirror_type& grid) {
    grad_calc_ = WLSGradient<T, host_exec_space, array_layout>(grid);
    grad_ = Vector3s<T, array_layout, host_mem_space>("GradVxAccess", grid.num_cells());
    grad_calc_.compute_gradients(grid, fs.vel.x(), grad_);
}

template <typename T>
Vector3<T> GradVxAccess<T>::access(
    const FlowStates<T, array_layout, host_mem_space>& fs,
    const IdealGas<T>& gas_model, const int i) {
    
    (void) gas_model;
    (void) fs;
    T grad_x = grad_.x(i);
    T grad_y = grad_.y(i);
    T grad_z = grad_.z(i);
    return Vector3<T>(grad_x, grad_y, grad_z);
}
template class GradVxAccess<double>;

template <typename T>
std::map<std::string, std::shared_ptr<ScalarAccessor<T>>> get_scalar_accessors() {
    return {
        {"pressure",
         std::shared_ptr<ScalarAccessor<T>>(new PressureAccess<T>())},
        {"temperature",
         std::shared_ptr<ScalarAccessor<T>>(new TemperatureAccess<T>())},
        {"density", std::shared_ptr<ScalarAccessor<T>>(new DensityAccess<T>())},
        {"energy",
         std::shared_ptr<ScalarAccessor<T>>(new InternalEnergyAccess<T>())},
        {"a", std::shared_ptr<ScalarAccessor<T>>(new SpeedOfSoundAccess<T>())},
        {"Mach",
         std::shared_ptr<ScalarAccessor<T>>(new MachNumberAccess<T>())}
    };
}
template std::map<std::string, std::shared_ptr<ScalarAccessor<double>>> 
get_scalar_accessors();

template <typename T>
std::map<std::string, std::shared_ptr<VectorAccessor<T>>> get_vector_accessors() {
    return {
        {"velocity",
         std::shared_ptr<VectorAccessor<T>>(new VelocityAccess<T>())}
    };
}
template std::map<std::string, std::shared_ptr<VectorAccessor<double>>> 
get_vector_accessors();
