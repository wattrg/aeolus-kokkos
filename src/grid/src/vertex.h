#ifndef VERTEX_H
#define VERTEX_H

#include "../../util/src/vector3.h"
#include "Kokkos_Core_fwd.hpp"

template <typename T>
struct Vertex {
public:
    Vertex(Vector3<T> pos) : _pos(pos) {}
    Vector3<T> &pos() { return _pos; }

    bool operator==(const Vertex<T> &other) const { return _pos == other._pos; }

private:
    Vector3<T> _pos;
};

template <typename T, class ExecSpace = Kokkos::DefaultExecutionSpace,
          class Layout = Kokkos::DefaultExecutionSpace::array_layout>
struct Vertices {
public:
    using array_layout = Layout;
    using memory_space = typename ExecSpace::memory_space;
    using vector_type = Vector3s<T, array_layout, memory_space>;
    using mirror_type =
        Vertices<T, Kokkos::DefaultHostExecutionSpace, array_layout>;

public:
    Vertices() {}

    Vertices(int num_vertices) {
        _positions = vector_type("Vertices", num_vertices);
    }

    void set_vertex_position(int vertex_id, Vector3<T> pos) {
        _positions(vertex_id, 0) = pos.x;
        _positions(vertex_id, 1) = pos.y;
        _positions(vertex_id, 2) = pos.z;
    }

    KOKKOS_INLINE_FUNCTION
    Vector3s<T, array_layout, memory_space> &positions() { return _positions; }

    KOKKOS_INLINE_FUNCTION
    const Vector3s<T, array_layout, memory_space> &positions() const {
        return _positions;
    }

    bool operator==(const Vertices &other) const {
        return _positions == other._positions;
    }

    KOKKOS_INLINE_FUNCTION
    int size() const { return _positions.size(); }

    mirror_type host_mirror() const { return mirror_type(_positions.size()); }

    template <class OtherSpace>
    void deep_copy(const Vertices<T, OtherSpace> &other) {
        _positions.deep_copy(other._positions);
    }

public:
    Vector3s<T, array_layout, memory_space> _positions;
};

#endif
