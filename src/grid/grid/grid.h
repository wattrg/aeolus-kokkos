#ifndef GRID_H
#define GRID_H

#include <grid/cell.h>
#include <grid/grid_io.h>
#include <grid/interface.h>

#include <limits>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

template <typename T, class ExecSpace = Kokkos::DefaultExecutionSpace,
          class Layout = Kokkos::DefaultExecutionSpace::array_layout>
class GridBlock {
public:
    using execution_space = ExecSpace;
    using memory_space = typename execution_space::memory_space;
    using array_layout = Layout;
    using host_execution_space = Kokkos::DefaultHostExecutionSpace;
    using host_mirror_mem_space = host_execution_space::memory_space;
    using mirror_type = GridBlock<T, host_execution_space, array_layout>;

public:
    GridBlock() {}

    GridBlock(const GridIO& grid_io, json& config) { init_grid_block(grid_io, config); }

    GridBlock(std::string file_name, json& config) {
        init_grid_block(GridIO(file_name), config);
    }

    GridBlock(
        Vertices<T, execution_space, array_layout> vertices,
        Interfaces<T, execution_space, array_layout> interfaces,
        Cells<T, execution_space, array_layout> cells, size_t dim, size_t num_valid_cells,
        size_t num_ghost_cells,
        std::map<std::string, Field<size_t, array_layout, memory_space>> ghost_cells,
        std::map<std::string, Field<size_t, array_layout, memory_space>> boundary_faces,
        std::vector<std::string> boundary_tags)
        : vertices_(vertices),
          interfaces_(interfaces),
          cells_(cells),
          dim_(dim),
          num_valid_cells_(num_valid_cells),
          num_ghost_cells_(num_ghost_cells),
          ghost_cells_(ghost_cells),
          boundary_faces_(boundary_faces),
          boundary_tags_(boundary_tags) {}

    GridBlock(size_t num_vertices, size_t num_faces, size_t num_valid_cells,
              size_t num_ghost_cells, size_t dim, size_t num_cell_vertex_ids,
              size_t num_face_vertex_ids, size_t num_face_ids,
              std::map<std::string, size_t> ghost_cell_sizes,
              std::map<std::string, size_t> boundary_face_sizes) {
        num_valid_cells_ = num_valid_cells;
        num_ghost_cells_ = num_ghost_cells;
        dim_ = dim;
        // int num_total_cells = num_valid_cells + num_ghost_cells;
        vertices_ = Vertices<T, execution_space, array_layout>(num_vertices);
        interfaces_ =
            Interfaces<T, execution_space, array_layout>(num_faces, num_face_vertex_ids);
        cells_ = Cells<T, execution_space, array_layout>(
            num_valid_cells, num_ghost_cells, num_cell_vertex_ids, num_face_ids);
        ghost_cells_ = std::map<std::string, Field<size_t, array_layout, memory_space>>{};
        boundary_faces_ =
            std::map<std::string, Field<size_t, array_layout, memory_space>>{};
        for (auto const& [key, val] : ghost_cell_sizes) {
            ghost_cells_.insert(
                {key, Field<size_t, array_layout, memory_space>("bc_cells", val)});
        }
        for (auto const& [key, val] : boundary_face_sizes) {
            boundary_faces_.insert(
                {key, Field<size_t, array_layout, memory_space>("bc_faces", val)});
        }
    }

    void init_grid_block(const GridIO& grid_io, json& config) {
        dim_ = grid_io.dim();
        json boundaries = config.at("boundaries");

        // set the positions of the vertices
        std::vector<Vertex<Ibis::real>> vertices = grid_io.vertices();
        vertices_ = Vertices<T, execution_space, array_layout>(vertices.size());
        auto host_vertices = vertices_.host_mirror();
        for (size_t i = 0; i < vertices.size(); i++) {
            host_vertices.set_vertex_position(i, vertices[i].pos());
        }
        vertices_.deep_copy(host_vertices);

        // some objects to assist in constructing the grid
        std::vector<std::vector<size_t>> interface_vertices{};
        std::vector<std::vector<size_t>> cell_vertices{};
        std::vector<std::vector<size_t>> cell_interface_ids{};
        InterfaceLookup interfaces = InterfaceLookup();

        // begin to assemble the interfaces and cells
        std::vector<ElemIO> cells = grid_io.cells();
        std::vector<ElemType> cell_shapes{};
        std::vector<ElemType> interface_shapes{};
        num_valid_cells_ = cells.size();
        for (size_t cell_i = 0; cell_i < cells.size(); cell_i++) {
            cell_vertices.push_back(cells[cell_i].vertex_ids());
            cell_shapes.push_back(cells[cell_i].cell_type());

            std::vector<ElemIO> cell_interfaces = cells[cell_i].interfaces();
            std::vector<size_t> cell_face_ids{};
            for (size_t face_i = 0; face_i < cell_interfaces.size(); face_i++) {
                std::vector<size_t> face_vertices = cell_interfaces[face_i].vertex_ids();

                // if this interface already exists, we use the existing one
                // if the interface doesn't exist, we make a new one
                size_t face_id = interfaces.id(face_vertices);
                if (face_id == std::numeric_limits<size_t>::max()) {
                    face_id = interfaces.insert(face_vertices);
                    interface_vertices.push_back(face_vertices);
                    interface_shapes.push_back(cell_interfaces[face_i].cell_type());
                }
                cell_face_ids.push_back(face_id);
            }
            cell_interface_ids.push_back(cell_face_ids);
        }

        std::map<size_t, size_t> ghost_cell_map =
            setup_boundaries(grid_io, boundaries, cell_vertices, interfaces, cell_shapes);

        interfaces_ = Interfaces<T, execution_space, array_layout>(interface_vertices,
                                                                   interface_shapes);

        cells_ = Cells<T, execution_space, array_layout>(
            cell_vertices, cell_interface_ids, cell_shapes, num_valid_cells_,
            num_ghost_cells_);

        // compute geometric and connectivity properties of the grid
        // The order these are done in is important -- some things
        // rely on other properties already being set
        interfaces_.compute_centres(vertices_);
        interfaces_.compute_areas(vertices_);
        interfaces_.compute_orientations(vertices_);
        cells_.compute_centroids(vertices_, interfaces_);
        compute_interface_connectivity(ghost_cell_map);
        cells_.compute_volumes(vertices_, interfaces_);
        compute_cell_neighbours();
        compute_ghost_cell_centres();
    }

    void compute_interface_connectivity(std::map<size_t, size_t> ghost_cells) {
        auto this_interfaces = interfaces_;
        auto this_cells = cells_;
        Kokkos::parallel_for(
            "compute_interface_connectivity",
            Kokkos::RangePolicy<execution_space>(0, num_valid_cells_),
            KOKKOS_LAMBDA(const size_t cell_i) {
                auto face_ids = this_cells.faces().face_ids(cell_i);
                T cell_x = this_cells.centroids().x(cell_i);
                T cell_y = this_cells.centroids().y(cell_i);
                T cell_z = this_cells.centroids().z(cell_i);
                for (size_t face_i = 0; face_i < face_ids.size(); face_i++) {
                    size_t face_id = face_ids[face_i];

                    // vector from the face centre to the cell centre
                    T dx = this_interfaces.centre().x(face_id) - cell_x;
                    T dy = this_interfaces.centre().y(face_id) - cell_y;
                    T dz = this_interfaces.centre().z(face_id) - cell_z;

                    // dot product of the vector from centre to centre with
                    // the interface normal vector
                    T dot = dx * this_interfaces.norm().x(face_id) +
                            dy * this_interfaces.norm().y(face_id) +
                            dz * this_interfaces.norm().z(face_id);
                    if (dot > 0.0) {
                        // cell is on the left of the face
                        this_interfaces.attach_cell_left(cell_i, face_id);
                        this_cells.faces().set_outsign(cell_i, face_i, 1);
                    } else {
                        // cell is on the right of face
                        this_interfaces.attach_cell_right(cell_i, face_id);
                        this_cells.faces().set_outsign(cell_i, face_i, -1);
                    }
                }
            });

        // TODO: loop through the ghost cells and attach them to
        // the other side of the interface.
        // Think about how to do this on the GPU
        auto interfaces_host = interfaces_.host_mirror();
        interfaces_host.deep_copy(interfaces_);
        for (auto boundary : ghost_cells) {
            int face_id = boundary.first;
            int ghost_cell_id = boundary.second;
            if (interfaces_host.left_cell(face_id) ==
                std::numeric_limits<size_t>::max()) {
                interfaces_host.attach_cell_left(ghost_cell_id, face_id);
            } else {
                interfaces_host.attach_cell_right(ghost_cell_id, face_id);
            }
        }
        interfaces_.deep_copy(interfaces_host);
    }

    mirror_type host_mirror() const {
        auto vertices = vertices_.host_mirror();
        auto interfaces = interfaces_.host_mirror();
        auto cells = cells_.host_mirror();
        std::map<std::string, Field<size_t, array_layout, host_mirror_mem_space>>
            ghost_cells{};
        std::map<std::string, Field<size_t, array_layout, host_mirror_mem_space>>
            boundary_faces{};

        for (auto const& [key, val] : ghost_cells_) {
            ghost_cells.insert({key, val.host_mirror()});
        }
        for (auto const& [key, val] : boundary_faces_) {
            boundary_faces.insert({key, val.host_mirror()});
        }

        return mirror_type(vertices, interfaces, cells, dim_, num_valid_cells_,
                           num_ghost_cells_, ghost_cells, boundary_faces, boundary_tags_);
    }

    template <class OtherSpace>
    void deep_copy(const GridBlock<T, OtherSpace, Layout>& other) {
        vertices_.deep_copy(other.vertices_);
        interfaces_.deep_copy(other.interfaces_);
        cells_.deep_copy(other.cells_);
        for (size_t i = 0; i < boundary_tags_.size(); i++) {
            std::string tag = boundary_tags_[i];
            ghost_cells_.at(tag).deep_copy(other.ghost_cells_.at(tag));
            boundary_faces_.at(tag).deep_copy(other.boundary_faces_.at(tag));
        }
    }

    bool operator==(const GridBlock& other) const {
        return (vertices_ == other.vertices_) && (interfaces_ == other.interfaces_) &&
               (cells_ == other.cells_);
    }

    KOKKOS_INLINE_FUNCTION
    Vertices<T, execution_space, array_layout>& vertices() { return vertices_; }

    KOKKOS_INLINE_FUNCTION
    const Vertices<T, execution_space, array_layout>& vertices() const {
        return vertices_;
    }

    KOKKOS_INLINE_FUNCTION
    size_t num_vertices() const { return vertices_.size(); }

    KOKKOS_INLINE_FUNCTION
    Interfaces<T, execution_space, array_layout>& interfaces() { return interfaces_; }

    KOKKOS_INLINE_FUNCTION
    const Interfaces<T, execution_space, array_layout>& interfaces() const {
        return interfaces_;
    }

    KOKKOS_INLINE_FUNCTION
    size_t num_interfaces() const { return interfaces_.size(); }

    KOKKOS_INLINE_FUNCTION
    Cells<T, execution_space, array_layout>& cells() { return cells_; }

    KOKKOS_INLINE_FUNCTION
    const Cells<T, execution_space, array_layout>& cells() const { return cells_; }

    KOKKOS_INLINE_FUNCTION
    size_t num_cells() const { return num_valid_cells_; }

    KOKKOS_INLINE_FUNCTION
    size_t num_ghost_cells() const { return num_ghost_cells_; }

    KOKKOS_INLINE_FUNCTION
    size_t num_total_cells() const { return num_valid_cells_ + num_ghost_cells_; }

    KOKKOS_INLINE_FUNCTION
    bool is_valid(const size_t i) const { return i < num_valid_cells_; }

    KOKKOS_INLINE_FUNCTION
    bool is_ghost(const size_t i) const { return i >= num_valid_cells_; }

    const Field<size_t, array_layout, memory_space>& boundary_faces(
        std::string boundary_tag) const {
        return boundary_faces_.at(boundary_tag);
    }

    const Field<size_t, array_layout, memory_space>& ghost_cells(
        std::string boundary_tag) const {
        return ghost_cells_.at(boundary_tag);
    }

    const std::vector<std::string>& boundary_tags() const { return boundary_tags_; }

    KOKKOS_INLINE_FUNCTION
    size_t dim() const { return dim_; }

    // this method requires the interface connectivity be set up correctly
    void compute_cell_neighbours() {
        auto this_interfaces = interfaces_;
        auto this_cells = cells_;
        Kokkos::parallel_for(
            "cell neighbours", num_cells(), KOKKOS_LAMBDA(const size_t cell_i) {
                auto cell_faces = this_cells.faces().face_ids(cell_i);
                for (size_t face_i = 0; face_i < cell_faces.size(); face_i++) {
                    size_t iface = cell_faces(face_i);
                    size_t neighbour;
                    size_t left_cell = this_interfaces.left_cell(iface);
                    if (left_cell == cell_i) {
                        neighbour = this_interfaces.right_cell(iface);
                    } else {
                        neighbour = left_cell;
                    }
                    this_cells.set_cell_neighbour(cell_i, face_i, neighbour);
                }
            });
    }

    // compute the cell centres of ghost cells by mirroring the cell
    // centre of the valid cell about the interface
    // needs to be called after setup_boundaries, compute_geometric_data,
    // and compute_interface_connectivity
    void compute_ghost_cell_centres() {
        auto this_interfaces = interfaces_;
        auto this_cells = cells_;
        size_t num_valid_cells = num_valid_cells_;
        for (auto& boundary : boundary_faces_) {
            auto boundary_faces = boundary_faces_[boundary.first];
            Kokkos::parallel_for(
                "ghost_cell_centres", boundary_faces.size(),
                KOKKOS_LAMBDA(const size_t face_i) {
                    // get the id of the cell to the left and right
                    // of this interface
                    size_t iface = boundary_faces(face_i);
                    size_t left_cell = this_interfaces.left_cell(iface);
                    size_t right_cell = this_interfaces.right_cell(iface);
                    size_t valid_cell;
                    size_t ghost_cell;
                    if (left_cell < num_valid_cells) {
                        valid_cell = left_cell;
                        ghost_cell = right_cell;
                    } else {
                        valid_cell = right_cell;
                        ghost_cell = left_cell;
                    }

                    // compute the vector from the valid cell centre to the
                    // centre of the interface
                    T face_x = this_interfaces.centre().x(iface);
                    T face_y = this_interfaces.centre().y(iface);
                    T face_z = this_interfaces.centre().z(iface);
                    T dx = face_x - this_cells.centroids().x(valid_cell);
                    T dy = face_y - this_cells.centroids().y(valid_cell);
                    T dz = face_z - this_cells.centroids().z(valid_cell);

                    // extrapolate the ghost cell centre
                    this_cells.centroids().x(ghost_cell) = face_x + dx;
                    this_cells.centroids().y(ghost_cell) = face_y + dy;
                    this_cells.centroids().z(ghost_cell) = face_z + dz;
                });
        }
    }

public:
    std::map<size_t, size_t> setup_boundaries(
        const GridIO& grid_io, json& boundaries,
        std::vector<std::vector<size_t>>& cell_vertices, InterfaceLookup& interfaces,
        std::vector<ElemType> cell_shapes) {
        (void)cell_vertices;
        (void)cell_shapes;
        num_ghost_cells_ = 0;
        std::map<size_t, size_t> ghost_cell_map;  // face_id -> ghost_cell_id
        for (auto bc : grid_io.bcs()) {
            // unpack the boundary data from the grid_io object
            std::string bc_label = bc.first;
            boundary_tags_.push_back(bc_label);
            std::vector<ElemIO> bc_faces = bc.second;
            json boundary_config = boundaries.at(bc_label);

            // loop over all the boundary faces for this boundary, keeping
            // track of which cells and faces belong to this boundary
            std::vector<size_t> ghost_cells{};
            std::vector<size_t> boundary_faces{};
            for (size_t boundary_i = 0; boundary_i < bc_faces.size(); boundary_i++) {
                size_t face_id = interfaces.id(bc_faces[boundary_i].vertex_ids());
                boundary_faces.push_back(face_id);
                if (boundary_config.at("ghost_cells") == true) {
                    size_t ghost_cell_id = num_valid_cells_ + num_ghost_cells_;
                    num_ghost_cells_++;
                    ghost_cells.push_back(ghost_cell_id);
                    ghost_cell_map.insert({face_id, ghost_cell_id});
                } else {
                    ghost_cell_map.insert({face_id, -1});  // no ghost cell
                }
            }

            // keep track of which faces/cells belong to
            // which boundary
            ghost_cells_.insert({bc_label, Field<size_t, array_layout, memory_space>(
                                               "bc_cells", ghost_cells)});
            boundary_faces_.insert({bc_label, Field<size_t, array_layout, memory_space>(
                                                  "bc_faces", boundary_faces)});
        }
        return ghost_cell_map;
    }

public:
    Vertices<T, execution_space, array_layout> vertices_;
    Interfaces<T, execution_space, array_layout> interfaces_;
    Cells<T, execution_space, array_layout> cells_;
    size_t dim_;
    size_t num_valid_cells_;
    size_t num_ghost_cells_;
    std::map<std::string, Field<size_t, array_layout, memory_space>> ghost_cells_;
    std::map<std::string, Field<size_t, array_layout, memory_space>> boundary_faces_;
    std::vector<std::string> boundary_tags_;
};

#endif
