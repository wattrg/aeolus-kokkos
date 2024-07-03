#ifndef VECTOR3_H
#define VECTOR3_H

#include <util/field.h>
#include <util/numeric_types.h>

#include <Kokkos_Core.hpp>

// A single vector with 3 components
template <typename T>
struct Vector3 {
    KOKKOS_INLINE_FUNCTION
    Vector3() : x(0.0), y(0.0), z(0.0) {}

    KOKKOS_INLINE_FUNCTION
    Vector3(T x) : x(x), y(0.0), z(0.0) {}

    KOKKOS_INLINE_FUNCTION
    Vector3(T x, T y) : x(x), y(y), z(0.0) {}

    KOKKOS_INLINE_FUNCTION
    Vector3(T x, T y, T z) : x(x), y(y), z(z) {}

    T x, y, z;

    bool operator==(const Vector3& other) const {
        return (std::fabs(x - other.x) < 1e-14) && (std::fabs(y - other.y) < 1e-14) &&
               (std::fabs(z - other.z) < 1e-14);
    }
};

template <typename T, class Layout = Kokkos::DefaultExecutionSpace::array_layout,
          class Space = Kokkos::DefaultExecutionSpace::memory_space>
struct Vector3s {
public:
    using view_type = Kokkos::View<T* [3], Layout, Space>;
    using array_layout = typename view_type::array_layout;
    using memory_space = typename view_type::memory_space;
    using mirror_view_type = typename view_type::host_mirror_type;
    using mirror_layout = typename mirror_view_type::array_layout;
    using mirror_space = typename mirror_view_type::memory_space;
    using mirror_type = Vector3s<T, mirror_layout, mirror_space>;

public:
    Vector3s() {}

    // ~Vector3s(){}

    Vector3s(std::string description, size_t n) { view_ = view_type(description, n); }

    Vector3s(size_t n) { view_ = view_type("Vector3s", n); }

    Vector3s(view_type data) : view_(data) {}

    KOKKOS_INLINE_FUNCTION
    T& operator()(const size_t i, const size_t j) { return view_(i, j); }

    KOKKOS_INLINE_FUNCTION
    T& operator()(const size_t i, const size_t j) const { return view_(i, j); }

    KOKKOS_INLINE_FUNCTION
    T& x(const size_t i) { return view_(i, 0); }

    KOKKOS_INLINE_FUNCTION
    T& x(const size_t i) const { return view_(i, 0); }

    KOKKOS_INLINE_FUNCTION
    auto x() { return Kokkos::subview(view_, Kokkos::ALL, 0); }

    KOKKOS_INLINE_FUNCTION
    auto x() const { return Kokkos::subview(view_, Kokkos::ALL, 0); }

    KOKKOS_INLINE_FUNCTION
    T& y(const size_t i) { return view_(i, 1); }

    KOKKOS_INLINE_FUNCTION
    T& y(const size_t i) const { return view_(i, 1); }

    KOKKOS_INLINE_FUNCTION
    auto y() { return Kokkos::subview(view_, Kokkos::ALL, 1); }

    KOKKOS_INLINE_FUNCTION
    auto y() const { return Kokkos::subview(view_, Kokkos::ALL, 1); }

    KOKKOS_INLINE_FUNCTION
    T& z(const size_t i) { return view_(i, 2); }

    KOKKOS_INLINE_FUNCTION
    T& z(const size_t i) const { return view_(i, 2); }

    KOKKOS_INLINE_FUNCTION
    auto z() { return Kokkos::subview(view_, Kokkos::ALL, 2); }

    KOKKOS_INLINE_FUNCTION
    auto z() const { return Kokkos::subview(view_, Kokkos::ALL, 2); }

    KOKKOS_INLINE_FUNCTION
    void set_vector(const Vector3<T>& vector, const size_t i) const {
        x(i) = vector.x;
        y(i) = vector.y;
        z(i) = vector.z;
    }

    KOKKOS_INLINE_FUNCTION
    Vector3<T> average_vectors(const size_t a, const size_t b) const {
        T x_avg = 0.5 * (x(a) + x(b));
        T y_avg = 0.5 * (y(a) + y(b));
        T z_avg = 0.5 * (z(a) + z(b));

        return Vector3<T>{x_avg, y_avg, z_avg};
    }

    KOKKOS_INLINE_FUNCTION
    size_t size() const { return view_.extent(0); }

    KOKKOS_INLINE_FUNCTION
    bool operator==(const Vector3s& other) const {
        if (this->size() != other.size()) {
            return false;
        }

        for (size_t i = 0; i < this->size(); i++) {
            if (fabs(view_(i, 0) - other.view_(i, 0)) > 1e-14) return false;
            if (fabs(view_(i, 1) - other.view_(i, 1)) > 1e-14) return false;
            if (fabs(view_(i, 2) - other.view_(i, 2)) > 1e-14) return false;
        }
        return true;
    }

    mirror_type host_mirror() const {
        auto mirror_view = Kokkos::create_mirror_view(view_);
        return mirror_type(mirror_view);
    }

    template <class OtherSpace>
    void deep_copy(const Vector3s<T, Layout, OtherSpace>& other) {
        Kokkos::deep_copy(view_, other.view_);
    }

public:
    view_type view_;
};

template <typename T>
KOKKOS_INLINE_FUNCTION T dot(const Vector3s<T>& a, const Vector3s<T>& b, const size_t i) {
    return a.x(i) * b.x(i) + a.y(i) * b.y(i) + a.z(i) * b.z(i);
}

template <typename T, class ExecSpace, class Layout>
KOKKOS_INLINE_FUNCTION void cross(const Vector3s<T, ExecSpace, Layout>& a,
                                  const Vector3s<T, ExecSpace, Layout>& b,
                                  const Vector3s<T, ExecSpace, Layout>& c,
                                  const size_t i) {
    c.x(i) = a.y(i) * b.z(i) - a.z(i) * b.y(i);
    c.y(i) = a.z(i) * b.x(i) - a.x(i) * b.z(i);
    c.z(i) = a.x(i) * b.y(i) - a.y(i) * b.x(i);
}

template <typename T>
void dot(const Vector3s<T>& a, const Vector3s<T>& b, Field<T>& result);

template <typename T>
void add(const Vector3s<T>& a, const Vector3s<T>& b, Vector3s<T>& result);

template <typename T>
void subtract(const Vector3s<T>& a, const Vector3s<T>& b, Vector3s<T>& result);

template <typename T>
void cross(const Vector3s<T>& a, const Vector3s<T>& b, Vector3s<T>& result);

template <typename T>
void scale_in_place(Vector3s<T>& a, T factor);

template <typename T>
void length(const Vector3s<T>& a, Field<T>& len);

template <typename T>
void normalise(Vector3s<T>& a);

template <typename T>
void transform_to_local_frame(Vector3s<T>& a, const Vector3s<T>& norm,
                              const Vector3s<T>& tan1, const Vector3s<T> tan2);

template <typename T>
void transform_to_global_frame(Vector3s<T>& a, const Vector3s<T>& norm,
                               const Vector3s<T>& tan1, const Vector3s<T>& tan2);

#endif
