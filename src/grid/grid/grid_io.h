#ifndef GRID_IO_H
#define GRID_IO_H

#include <grid/vertex.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

enum class GridFileType {
    Native,
    Su2,
};

enum class ElemType {
    Line,
    Tri,
    Tetra,
    Quad,
    Hex,
    Wedge,
    Pyramid,
};

enum class FaceOrder {
    Vtk,
};

struct ElemIO {
    ElemIO() {}

    ElemIO(std::vector<size_t> ids, ElemType type, FaceOrder face_order)
        : vertex_ids_(ids), cell_type_(type), face_order_(face_order) {}

    ElemIO(const ElemIO &other);

    ElemIO &operator=(const ElemIO &other);

    bool operator==(const ElemIO &other) const {
        return (vertex_ids_ == other.vertex_ids_) && (cell_type_ == other.cell_type_);
    }

    std::vector<size_t> vertex_ids() const { return vertex_ids_; }

    ElemType cell_type() const { return cell_type_; }

    FaceOrder face_order() const { return face_order_; }

    std::vector<ElemIO> interfaces() const;

    friend std::ostream &operator<<(std::ostream &file, const ElemIO &elem_io);

private:
    std::vector<size_t> vertex_ids_{};
    ElemType cell_type_;
    FaceOrder face_order_;
};

struct CellMapping {
    size_t local_cell;
    size_t other_block;
    size_t other_cell;

    CellMapping(size_t local_cell_, size_t other_block_, size_t other_cell_)
        : local_cell(local_cell_), other_block(other_block_), other_cell(other_cell_) {}

    bool operator==(const CellMapping &other) const {
        return (local_cell == other.local_cell) && (other_block == other.other_block) &&
               (other_cell == other.other_cell);
    }

    friend std::ostream &operator<<(std::ostream &file, const CellMapping &map);
};

struct GridIO {
public:
    GridIO(std::vector<Vertex<Ibis::real>> vertices, std::vector<ElemIO> cells,
           std::unordered_map<std::string, std::vector<ElemIO>> markers, int dim)
        : vertices_(vertices), cells_(cells), markers_(markers), dim_(dim) {}

    GridIO(std::vector<Vertex<Ibis::real>> vertices, std::vector<ElemIO> cells,
           std::unordered_map<std::string, std::vector<ElemIO>> markers)
        : vertices_(vertices), cells_(cells), markers_(markers) {}

    GridIO(std::string file_name);

    GridIO(const GridIO &monolithic_grid, const std::vector<size_t> &cells_to_include,
           const std::vector<CellMapping> &&cell_mapping, size_t id);

    GridIO() {}

    bool operator==(const GridIO &other) const {
        return (vertices_ == other.vertices_) && (cells_ == other.cells_) &&
               (markers_ == other.markers_) && (cell_mapping_ == other.cell_mapping_);
    }

    std::vector<Vertex<Ibis::real>> vertices() const { return vertices_; }

    std::vector<ElemIO> cells() const { return cells_; }

    std::unordered_map<std::string, std::vector<ElemIO>> markers() const {
        return markers_;
    }

    std::vector<CellMapping> cell_mapping() const { return cell_mapping_; }

    size_t dim() const { return dim_; }

    size_t id() const { return id_; }

    void read_su2_grid(std::istream &grid_file);
    void write_su2_grid(std::ostream &grid_file);

    void read_mapped_cells(std::istream &file);
    void write_mapped_cells(std::ostream &file);

private:
    std::vector<Vertex<Ibis::real>> vertices_{};
    std::vector<ElemIO> cells_{};
    std::unordered_map<std::string, std::vector<ElemIO>> markers_;
    size_t dim_;
    size_t id_ = 0;

    // for partitioned grids, we need to know which cells connect
    // to cells in a different block.
    std::vector<CellMapping> cell_mapping_;
};

#endif
