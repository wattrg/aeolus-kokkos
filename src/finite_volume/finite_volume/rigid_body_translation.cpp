#include <finite_volume/rigid_body_translation.h>

template <typename T>
RigidBodyTranslation<T>::RigidBodyTranslation(json config) {
    json velocity = config.at("velocity");
    Ibis::real x = velocity.at("x");
    Ibis::real y = velocity.at("y");
    Ibis::real z = velocity.at("z");
    vel_ = Vector3<T>(x, y, z);
}

template <typename T>
void RigidBodyTranslation<T>::compute_vertex_velocities(const FlowStates<T>& fs,
                                                        const GridBlock<T>& grid,
                                                        Vector3s<T> vertex_vel) {
    (void)fs;
    (void)grid;
    Vector3<T> vel = vel_;
    Kokkos::parallel_for(
        "shock_fitting", grid.num_vertices(), KOKKOS_LAMBDA(const size_t i) {
            vertex_vel.x(i) = T(vel.x);
            vertex_vel.y(i) = T(vel.y);
            vertex_vel.z(i) = T(vel.z);
        });
}
template class RigidBodyTranslation<Ibis::real>;
template class RigidBodyTranslation<Ibis::dual>;
