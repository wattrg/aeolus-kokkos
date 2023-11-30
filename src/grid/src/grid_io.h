#ifndef GRID_IO_H
#define GRID_IO_H

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "vertex.h"

enum class GridFileType {
    Native,
    Su2,
};

enum class ElemType {
    Line,
    Tri,
    Quad,
    Hex,
    Wedge,
    Pyramid,
};

enum class FaceOrder {
    Vtk,
};

struct ElemIO {
    ElemIO(std::vector<int> ids, ElemType type, FaceOrder face_order)
        : vertex_ids_(ids), cell_type_(type), face_order_(face_order) {}

    bool operator==(const ElemIO &other) const {
        return (vertex_ids_ == other.vertex_ids_) &&
               (cell_type_ == other.cell_type_);
    }

    std::vector<int> vertex_ids() const { return vertex_ids_; }

    ElemType cell_type() const { return cell_type_; }

    std::vector<ElemIO> interfaces() const;

private:
    std::vector<int> vertex_ids_{};
    ElemType cell_type_;
    FaceOrder face_order_;
};

struct GridIO {
public:
    GridIO(std::vector<Vertex<double>> vertices, std::vector<ElemIO> cells,
           std::unordered_map<std::string, std::vector<ElemIO>> bcs)
        : vertices_(vertices), cells_(cells), bcs_(bcs) {}

    GridIO(std::string file_name);

    bool operator==(const GridIO &other) const {
        return (vertices_ == other.vertices_) && (cells_ == other.cells_) &&
               (bcs_ == other.bcs_);
    }

    std::vector<Vertex<double>> vertices() const { return vertices_; }

    std::vector<ElemIO> cells() const { return cells_; }

    std::unordered_map<std::string, std::vector<ElemIO>> bcs() const {
        return bcs_;
    }

    int dim() const { return dim_; }

private:
    std::vector<Vertex<double>> vertices_{};
    std::vector<ElemIO> cells_{};
    std::unordered_map<std::string, std::vector<ElemIO>> bcs_;
    int dim_;

    void _read_su2_grid(std::ifstream &grid_file);
};

#endif
