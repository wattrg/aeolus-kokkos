#ifndef INTERFACE_H
#define INTERFACE_H

// #include <grid/cell.h>
#include <grid/geom.h>
#include <grid/grid_io.h>
#include <grid/vertex.h>
#include <util/field.h>
#include <util/ragged_array.h>
#include <util/vector3.h>

#include <limits>
#include <unordered_map>

template <typename T, class ExecSpace = Kokkos::DefaultExecutionSpace,
          class Layout = Kokkos::DefaultExecutionSpace::array_layout>
struct Interfaces {
public:
    using execution_space = ExecSpace;
    using memory_space = typename execution_space::memory_space;
    using array_layout = Layout;
    using host_execution_space = Kokkos::DefaultHostExecutionSpace;
    using vector_type = Vector3s<T, array_layout, memory_space>;
    using host_mirror_mem_space = Kokkos::DefaultHostExecutionSpace::memory_space;
    using mirror_type = Interfaces<T, host_execution_space, array_layout>;

    // I have no idea why it is execution_space and not memory_space
    // here. But execution_space works and memory_space doesn't...
    using id_type = Ibis::RaggedArray<size_t, array_layout, execution_space>;

public:
    Interfaces() {}

    Interfaces(std::vector<std::vector<size_t>> ids, std::vector<ElemType> shapes) {
        vertex_ids_ = id_type(ids);
        size_ = vertex_ids_.num_rows();
        shape_ = Field<ElemType, array_layout, memory_space>("Interface::shape",
                                                             shapes.size());
        auto shape_mirror = shape_.host_mirror();
        for (size_t i = 0; i < size_; i++) {
            shape_mirror(i) = shapes[i];
        }
        shape_.deep_copy(shape_mirror);

        // geometry
        norm_ = vector_type("Interface::norm", size_);
        tan1_ = vector_type("Interface::tan1", size_);
        tan2_ = vector_type("Interface::tan2", size_);
        centre_ = vector_type("Interface::centre", size_);
        area_ = Field<T, array_layout, memory_space>("Interface::area", size_);

        // set left and right cells to maximum integer to indicate they haven't
        // been connected up to any cells yet
        left_cells_ = Field<size_t, array_layout, memory_space>("Interface::left", size_);
        right_cells_ =
            Field<size_t, array_layout, memory_space>("Interface::right", size_);
        left_cells_.deep_copy(std::numeric_limits<size_t>::max());
        right_cells_.deep_copy(std::numeric_limits<size_t>::max());
    }

    Interfaces(size_t num_interfaces, size_t num_vertex_ids) {
        vertex_ids_ = id_type(num_vertex_ids, num_interfaces);
        size_ = num_interfaces;
        shape_ = Field<ElemType, array_layout, memory_space>("Interface::shape", size_);
        norm_ = vector_type("Interface::norm", size_);
        tan1_ = vector_type("Interface::tan1", size_);
        tan2_ = vector_type("Interface::tan2", size_);
        centre_ = vector_type("Interface::centre", size_);
        area_ = Field<T, array_layout, memory_space>("Interface::area", size_);
        left_cells_ = Field<size_t, array_layout, memory_space>("Interface::left", size_);
        right_cells_ =
            Field<size_t, array_layout, memory_space>("Interface::right", size_);
    }

    Interfaces(id_type vertex_ids, Field<size_t, array_layout, memory_space> left_cells,
               Field<size_t, array_layout, memory_space> right_cells,
               Field<T, array_layout, memory_space> area,
               Field<ElemType, array_layout, memory_space> shape, vector_type norm,
               vector_type tan1, vector_type tan2, vector_type centre)
        : size_(vertex_ids.num_rows()),
          vertex_ids_(vertex_ids),
          left_cells_(left_cells),
          right_cells_(right_cells),
          area_(area),
          shape_(shape),
          norm_(norm),
          tan1_(tan1),
          tan2_(tan2),
          centre_(centre) {}

    mirror_type host_mirror() const {
        auto vertex_ids = vertex_ids_.host_mirror();
        auto left_cells = left_cells_.host_mirror();
        auto right_cells = right_cells_.host_mirror();
        auto area = area_.host_mirror();
        auto shape = shape_.host_mirror();
        auto norm = norm_.host_mirror();
        auto tan1 = tan1_.host_mirror();
        auto tan2 = tan2_.host_mirror();
        auto centre = centre_.host_mirror();
        return mirror_type(vertex_ids, left_cells, right_cells, area, shape, norm, tan1,
                           tan2, centre);
    }

    template <class OtherSpace>
    void deep_copy(const Interfaces<T, OtherSpace, array_layout>& other) {
        vertex_ids_.deep_copy(other.vertex_ids_);
        shape_.deep_copy(other.shape_);
        norm_.deep_copy(other.norm_);
        tan1_.deep_copy(other.tan1_);
        tan2_.deep_copy(other.tan2_);
        centre_.deep_copy(other.centre_);
        area_.deep_copy(other.area_);
        left_cells_.deep_copy(other.left_cells_);
        right_cells_.deep_copy(other.right_cells_);
    }

    bool operator==(const Interfaces& other) const {
        return vertex_ids_ == other.vertex_ids_;
    }

    KOKKOS_INLINE_FUNCTION
    id_type& vertex_ids() { return vertex_ids_; }

    KOKKOS_INLINE_FUNCTION
    const id_type& vertex_ids() const { return vertex_ids_; }

    KOKKOS_INLINE_FUNCTION
    const Field<T, array_layout, memory_space>& area() const { return area_; }

    KOKKOS_INLINE_FUNCTION
    T& area(size_t i) const { return area_(i); }

    KOKKOS_INLINE_FUNCTION
    T& area(size_t i) { return area_(i); }

    KOKKOS_INLINE_FUNCTION
    Vector3s<T, array_layout, memory_space>& norm() { return norm_; }

    KOKKOS_INLINE_FUNCTION
    const vector_type& norm() const { return norm_; }

    KOKKOS_INLINE_FUNCTION
    vector_type& tan1() { return tan1_; }

    KOKKOS_INLINE_FUNCTION
    const vector_type& tan1() const { return tan1_; }

    KOKKOS_INLINE_FUNCTION
    vector_type& tan2() { return tan2_; }

    KOKKOS_INLINE_FUNCTION
    const vector_type& tan2() const { return tan2_; }

    KOKKOS_INLINE_FUNCTION
    vector_type& centre() { return centre_; }

    KOKKOS_INLINE_FUNCTION
    const vector_type& centre() const { return centre_; }

    KOKKOS_INLINE_FUNCTION
    void attach_cell_left(const size_t cell_id, const size_t face_id) const {
        left_cells_(face_id) = cell_id;
    }

    KOKKOS_INLINE_FUNCTION
    void attach_cell_right(const size_t cell_id, const size_t face_id) const {
        right_cells_(face_id) = cell_id;
    }

    KOKKOS_INLINE_FUNCTION
    size_t left_cell(const size_t face_id) const { return left_cells_(face_id); }

    KOKKOS_INLINE_FUNCTION
    size_t right_cell(const size_t face_id) const { return right_cells_(face_id); }

    KOKKOS_INLINE_FUNCTION
    size_t size() const { return vertex_ids_.num_rows(); }

    void compute_orientations(Vertices<T, execution_space> vertices) {
        // set the face tangents in parallel
        auto this_norm = norm_;
        auto this_tan1 = tan1_;
        auto this_tan2 = tan2_;
        auto this_vertex_ids = vertex_ids_;
        auto shape = shape_;
        Kokkos::parallel_for(
            "Interfaces::compute_orientations",
            Kokkos::RangePolicy<execution_space>(0, norm_.size()),
            KOKKOS_LAMBDA(const size_t i) {
                auto vertex_ids = this_vertex_ids(i);
                T x0 = vertices.positions().x(vertex_ids(0));
                T x1 = vertices.positions().x(vertex_ids(1));
                T y0 = vertices.positions().y(vertex_ids(0));
                T y1 = vertices.positions().y(vertex_ids(1));
                T z0 = vertices.positions().z(vertex_ids(0));
                T z1 = vertices.positions().z(vertex_ids(1));
                T ilength =
                    1. / Ibis::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) +
                                    (z1 - z0) * (z1 - z0));
                this_tan1.x(i) = ilength * (x1 - x0);
                this_tan1.y(i) = ilength * (y1 - y0);
                this_tan1.z(i) = ilength * (z1 - z0);

                switch (shape(i)) {
                    case ElemType::Line: {
                        auto vertex_ids = this_vertex_ids(i);
                        this_tan2.x(i) = 0.0;
                        this_tan2.y(i) = 0.0;
                        this_tan2.z(i) = 1.0;
                        break;
                    }
                    case ElemType::Tri:
                    case ElemType::Quad: {
                        auto vertex_ids = this_vertex_ids(i);

                        // The unit vector in the direction from vertex 0 to vertex 1
                        T t1x = this_tan1.x(i);
                        T t1y = this_tan1.y(i);
                        T t1z = this_tan1.z(i);

                        // The vector in the direction from vertex 0 to vertex 2 (t2star)
                        T t2xstar = vertices.positions().x(vertex_ids(2)) -
                                    vertices.positions().x(vertex_ids(0));
                        T t2ystar = vertices.positions().y(vertex_ids(2)) -
                                    vertices.positions().y(vertex_ids(0));
                        T t2zstar = vertices.positions().z(vertex_ids(2)) -
                                    vertices.positions().z(vertex_ids(0));

                        // project t2star onto t1; the direction we're after
                        // is the difference between the projection and the
                        // actual vector from 0->2
                        T proj = t1x * t2xstar + t1y * t2ystar + t1z * t2zstar;
                        T t2x = t2xstar - proj * t1x;
                        T t2y = t2ystar - proj * t1y;
                        T t2z = t2zstar - proj * t1z;

                        // normalise the vector
                        T ilength = 1.0 / Ibis::sqrt(t2x * t2x + t2y * t2y + t2z * t2z);
                        this_tan2.x(i) = ilength * t2x;
                        this_tan2.y(i) = ilength * t2y;
                        this_tan2.z(i) = ilength * t2z;
                        break;
                    }
                    default:
                        printf("Invalid interface shape");
                        break;
                }
                cross(this_tan1, this_tan2, this_norm, i);
            });
    }

    void compute_areas(Vertices<T, execution_space> vertices) {
        auto this_area = area_;
        auto shape = shape_;
        auto this_vertex_ids = vertex_ids_;
        Kokkos::parallel_for(
            "Interfaces::compute_areas",
            Kokkos::RangePolicy<execution_space>(0, area_.size()),
            KOKKOS_LAMBDA(const size_t i) {
                switch (shape(i)) {
                    case ElemType::Line: {
                        auto vertex_ids = this_vertex_ids(i);
                        this_area(i) = Ibis::distance_between_points(
                            vertices.positions(), vertex_ids(0), vertex_ids(1));
                        break;
                    }
                    case ElemType::Tri: {
                        auto vertex_ids = this_vertex_ids(i);
                        this_area(i) =
                            Ibis::area_of_triangle(vertices.positions(), vertex_ids(0),
                                                   vertex_ids(1), vertex_ids(2));
                        break;
                    }
                    case ElemType::Quad: {
                        auto vertex_ids = this_vertex_ids(i);
                        this_area(i) = Ibis::area_of_quadrilateral(
                            vertices.positions(), vertex_ids(0), vertex_ids(1),
                            vertex_ids(2), vertex_ids(3));
                        break;
                    }
                    case ElemType::Tetra:
                        printf("Invalid interface");
                        break;
                    case ElemType::Hex: {
                        printf("Invalid interface");
                        break;
                    }
                    case ElemType::Wedge: {
                        printf("Invalid interface");
                        break;
                    }
                    case ElemType::Pyramid: {
                        printf("Invalid interface");
                        break;
                    }
                }
            });
    }

    void compute_centres(Vertices<T, execution_space> vertices) {
        auto centre = centre_;
        auto vertex_ids = vertex_ids_;
        Kokkos::parallel_for(
            "Interfaces::compute_centres",
            Kokkos::RangePolicy<execution_space>(0, centre_.size()),
            KOKKOS_LAMBDA(const size_t face_i) {
                auto face_vertices = vertex_ids(face_i);
                T x = 0.0;
                T y = 0.0;
                T z = 0.0;
                size_t num_vertices = face_vertices.size();
                for (size_t vtx_i = 0; vtx_i < num_vertices; vtx_i++) {
                    size_t vtx_id = face_vertices[vtx_i];
                    x += vertices.positions().x(vtx_id);
                    y += vertices.positions().y(vtx_id);
                    z += vertices.positions().z(vtx_id);
                }
                centre.x(face_i) = x / num_vertices;
                centre.y(face_i) = y / num_vertices;
                centre.z(face_i) = z / num_vertices;
            });
    }

public:
    size_t size_;

    // the id's of the vertices forming each interface
    id_type vertex_ids_;

    // the cells to the left/right of the interface
    Field<size_t, array_layout, memory_space> left_cells_;
    Field<size_t, array_layout, memory_space> right_cells_;

    // geometric data
    Field<T, array_layout, memory_space> area_;
    Field<ElemType, array_layout, memory_space> shape_;
    vector_type norm_;
    vector_type tan1_;
    vector_type tan2_;
    vector_type centre_;
};

// Efficient look-up of interface ID
// from the index of the vertices
// forming the interface
struct InterfaceLookup {
public:
    InterfaceLookup();

    size_t insert(std::vector<size_t> vertex_ids);
    bool contains(std::vector<size_t> vertex_ids);
    size_t id(std::vector<size_t> vertex_ids);

private:
    std::unordered_map<std::string, size_t> hash_map_;

    std::string hash_vertex_ids(std::vector<size_t> vertex_ids);
    bool contains_hash(std::string hash);
};

#endif
