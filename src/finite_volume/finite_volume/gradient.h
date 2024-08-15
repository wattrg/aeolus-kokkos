#ifndef GRADIENT_H
#define GRADIENT_H

#include <grid/grid.h>
#include <util/ragged_array.h>

#include <Kokkos_Core.hpp>

template <typename T, class Layout = Kokkos::DefaultExecutionSpace::array_layout,
          class Space = Kokkos::DefaultExecutionSpace::memory_space>
struct Gradients {
    Gradients() {}

    Gradients(int num_cells, bool need_pressure, bool need_temp, bool need_u,
              bool need_rho, bool viscous) {
        bool high_order = (need_pressure || need_temp || need_u || need_rho);
        if (high_order || viscous) {
            vx = Vector3s<T, Layout, Space>("Gradients::vx", num_cells);
            vy = Vector3s<T, Layout, Space>("Gradients::vy", num_cells);
            vz = Vector3s<T, Layout, Space>("Gradients::vz", num_cells);
        }
        if (need_pressure) {
            p = Vector3s<T, Layout, Space>("Gradients::p", num_cells);
        }
        if (need_u) {
            u = Vector3s<T, Layout, Space>("Gradients::u", num_cells);
        }
        if (need_rho) {
            rho = Vector3s<T, Layout, Space>("Gradients::rho", num_cells);
        }
        if (need_temp || viscous) {
            temp = Vector3s<T, Layout, Space>("Gradients::T", num_cells);
        }
    }

    Vector3s<T, Layout, Space> p;
    Vector3s<T, Layout, Space> rho;
    Vector3s<T, Layout, Space> u;
    Vector3s<T, Layout, Space> temp;
    Vector3s<T, Layout, Space> vx;
    Vector3s<T, Layout, Space> vy;
    Vector3s<T, Layout, Space> vz;
};

template <typename T, class ExecSpace = Kokkos::DefaultExecutionSpace,
          class Layout = Kokkos::DefaultExecutionSpace::array_layout>
class WLSGradient {
public:
    using memory_space = typename ExecSpace::memory_space;
    using view_type = Kokkos::View<T**, Layout, memory_space>;
    using HostMirror = WLSGradient<T, Kokkos::DefaultHostExecutionSpace, Layout>;

public:
    WLSGradient() {}

    WLSGradient(const GridBlock<T, ExecSpace, Layout>& block) {
        int num_cells = block.num_cells();
        int num_rs = block.dim() == 2 ? 3 : 6;
        r_ = Kokkos::View<T**, Layout, memory_space>("WLSGradient::r", num_cells, num_rs);
        compute_workspace_(block);
    }

    WLSGradient(view_type rs) : r_(rs) {}

    template <class SubView>
    void compute_gradients(const GridBlock<T, ExecSpace, Layout>& block,
                           const SubView values, Vector3s<T, Layout, memory_space> grad) {
        auto cells = block.cells();
        int dim = block.dim();
        Kokkos::parallel_for(
            "WLSGradient::compute_gradients", block.num_cells(),
            KOKKOS_CLASS_LAMBDA(const int i) {
                auto neighbours = cells.neighbour_cells(i);
                T grad_x_ = 0.0;
                T grad_y_ = 0.0;
                T grad_z_ = 0.0;
                T u_i = values(i);
                T r11 = r_11_(i);
                T r12 = r_12_(i);
                T r22 = r_22_(i);
                T r13 = 0.0;
                T r23 = 0.0;
                T r33 = 0.0;
                if (dim == 3) {
                    r13 = r_13_(i);
                    r23 = r_23_(i);
                    r33 = r_33_(i);
                }
                T beta = (r12 * r23 - r13 * r23) / (r11 * r22);
                T xi = cells.centroids().x(i);
                T yi = cells.centroids().y(i);
                T zi = cells.centroids().z(i);
                for (unsigned int j = 0; j < neighbours.size(); j++) {
                    int neighbour_j = neighbours(j);
                    T diff_u = values(neighbour_j) - u_i;
                    T dx = cells.centroids().x(neighbour_j) - xi;
                    T dy = cells.centroids().y(neighbour_j) - yi;
                    T dz = cells.centroids().z(neighbour_j) - zi;
                    T alpha_1 = dx / (r11 * r11);
                    T alpha_2 = 1.0 / (r22 * r22) * (dy - r12 / r11 * dx);
                    T alpha_3 = 0.0;
                    if (dim == 3) {
                        alpha_3 = 1.0 / (r33 * r33) * (dz - r23 / r22 * dy + beta * dx);
                    }
                    T w_1 = alpha_1 - r12 / r11 * alpha_2 + beta * alpha_3;
                    T w_2 = alpha_2 - r23 / r22 * alpha_3;
                    T w_3 = alpha_3;
                    grad_x_ += w_1 * diff_u;
                    grad_y_ += w_2 * diff_u;
                    grad_z_ += w_3 * diff_u;
                }
                grad.x(i) = grad_x_;
                grad.y(i) = grad_y_;
                grad.z(i) = grad_z_;
            });
    }

public:
    void compute_workspace_(const GridBlock<T, ExecSpace, Layout>& block) {
        auto cells = block.cells();
        int dim = block.dim();
        Kokkos::parallel_for(
            "WLSGradient::compute_workspace_::r", block.num_cells(),
            KOKKOS_CLASS_LAMBDA(const int i) {
                auto neighbours = cells.neighbour_cells(i);
                T sum_dxdx = 0.0;
                T sum_dxdy = 0.0;
                T sum_dxdz = 0.0;
                T sum_dydy = 0.0;
                T sum_dydz = 0.0;
                T sum_dzdz = 0.0;
                T xi = cells.centroids().x(i);
                T yi = cells.centroids().y(i);
                T zi = cells.centroids().z(i);
                for (unsigned int j = 0; j < neighbours.size(); j++) {
                    int neighbour_j = neighbours(j);
                    T dx = cells.centroids().x(neighbour_j) - xi;
                    T dy = cells.centroids().y(neighbour_j) - yi;
                    T dz = cells.centroids().z(neighbour_j) - zi;
                    sum_dxdx += dx * dx;
                    sum_dxdy += dx * dy;
                    sum_dxdz += dx * dz;
                    sum_dydy += dy * dy;
                    sum_dydz += dy * dz;
                    sum_dzdz += dz * dz;
                }
                T r11 = Ibis::sqrt(sum_dxdx);
                T r12 = 1.0 / r11 * sum_dxdy;
                T r22 = Ibis::sqrt(sum_dydy - r12 * r12);
                T r13 = 1.0 / r11 * sum_dxdz;
                T r23 = 1.0 / r22 * (sum_dydz - r12 / r11 * sum_dxdz);
                T r33 = Ibis::sqrt(sum_dzdz - (r13 * r13 + r23 * r23));
                r_11_(i) = r11;
                r_12_(i) = r12;
                r_22_(i) = r22;
                if (dim == 3) {
                    r_13_(i) = r13;
                    r_23_(i) = r23;
                    r_33_(i) = r33;
                }
            });
    }

public:
    HostMirror host_mirror() const {
        auto r_mirror = Kokkos::create_mirror_view(r_);
        return HostMirror(r_mirror);
    }

    template <class OtherSpace>
    void deep_copy(const WLSGradient<T, OtherSpace, Layout>& other) {
        Kokkos::deep_copy(r_, other.r_);
    }

public:
    view_type r_;

public:
    KOKKOS_INLINE_FUNCTION
    T& r_11_(const int cell_i) { return r_(cell_i, 0); }

    KOKKOS_INLINE_FUNCTION
    T& r_11_(const int cell_i) const { return r_(cell_i, 0); }

    KOKKOS_INLINE_FUNCTION
    T& r_12_(const int cell_i) { return r_(cell_i, 1); }

    KOKKOS_INLINE_FUNCTION
    T& r_12_(const int cell_i) const { return r_(cell_i, 1); }

    KOKKOS_INLINE_FUNCTION
    T& r_22_(const int cell_i) { return r_(cell_i, 2); }

    KOKKOS_INLINE_FUNCTION
    T& r_22_(const int cell_i) const { return r_(cell_i, 2); }

    KOKKOS_INLINE_FUNCTION
    T& r_13_(const int cell_i) { return r_(cell_i, 3); }

    KOKKOS_INLINE_FUNCTION
    T& r_13_(const int cell_i) const { return r_(cell_i, 3); }

    KOKKOS_INLINE_FUNCTION
    T& r_23_(const int cell_i) { return r_(cell_i, 4); }

    KOKKOS_INLINE_FUNCTION
    T& r_23_(const int cell_i) const { return r_(cell_i, 4); }

    KOKKOS_INLINE_FUNCTION
    T& r_33_(const int cell_i) { return r_(cell_i, 5); }

    KOKKOS_INLINE_FUNCTION
    T& r_33_(const int cell_i) const { return r_(cell_i, 5); }
};

#endif
