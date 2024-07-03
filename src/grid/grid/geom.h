#ifndef GEOM_H
#define GEOM_H

#include <util/vector3.h>

namespace Ibis {

// Compute the distance from point i to point j
// The i and j are indices into `positions`, which contains
// the actual positions of all the points
template <typename T, class Layout, class Space>
KOKKOS_INLINE_FUNCTION T distance_between_points(
    const Vector3s<T, Layout, Space>& positions, const size_t i, const size_t j) {
    T xi = positions.x(i);
    T xj = positions.x(j);
    T yi = positions.y(i);
    T yj = positions.y(j);
    T zi = positions.z(i);
    T zj = positions.z(j);

    T dx = xj - xi;
    T dy = yj - yi;
    T dz = zj - zi;

    return Ibis::sqrt(dx * dx + dy * dy + dz * dz);
}

// calculate the area of a triangle with vertices a, b, and c
template <typename T, class Layout, class Space>
KOKKOS_INLINE_FUNCTION T area_of_triangle(const Vector3s<T, Layout, Space>& pos,
                                          const size_t a, const size_t b,
                                          const size_t c) {
    // vector from a -> b
    T ab_x = pos.x(b) - pos.x(a);
    T ab_y = pos.y(b) - pos.y(a);
    T ab_z = pos.z(b) - pos.z(a);

    // vector from a -> c
    T ac_x = pos.x(c) - pos.x(a);
    T ac_y = pos.y(c) - pos.y(a);
    T ac_z = pos.z(c) - pos.z(a);

    // cross product of the two vectors
    T cross_x = ab_y * ac_z - ab_z * ac_y;
    T cross_y = ab_z * ac_x - ab_x * ac_z;
    T cross_z = ab_x * ac_y - ab_y * ac_x;

    return 0.5 * Ibis::sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
}

// calculate the area of a quadrilateral with vertices
// a, b, c, d
// this computation could be bad if the quadrilateral
// in question is concave...
template <typename T, class Layout, class Space>
KOKKOS_INLINE_FUNCTION T area_of_quadrilateral(const Vector3s<T, Layout, Space>& pos,
                                               const size_t a, const size_t b,
                                               const size_t c, const size_t d) {
    T a1 = area_of_triangle(pos, a, b, c);
    T a2 = area_of_triangle(pos, a, c, d);
    return a1 + a2;
}

// Compute the volume of a generic cell using Eq. 5.15 of
// Blazek's book
template <typename T, class Layout, class Space, class OutSubView, class FaceSubView>
KOKKOS_INLINE_FUNCTION T volume_of_generic_cell(
    const Vector3s<T, Space, Layout>& face_cntr,
    const Vector3s<T, Space, Layout>& face_norm, const Field<T, Space, Layout>& face_area,
    OutSubView outsign, FaceSubView face_ids) {
    size_t n_faces = face_ids.extent(0);

    T volume = 0.0;
    for (size_t i = 0; i < n_faces; i++) {
        size_t face_id = face_ids(i);
        int out = outsign(i);
        T dot = face_cntr.x(face_id) * out * face_norm.x(face_id) +
                face_cntr.y(face_id) * out * face_norm.y(face_id) +
                face_cntr.z(face_id) * out * face_norm.z(face_id);
        volume += dot * face_area(face_id);
    }
    return volume / 3.0;
}

}  // namespace Ibis

#endif
