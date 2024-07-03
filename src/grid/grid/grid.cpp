
#include <doctest/doctest.h>
#include <grid/grid.h>
#include <grid/grid_io.h>
#include <grid/interface.h>

struct GridInfo {
    Vertices<double, Kokkos::DefaultHostExecutionSpace> vertices;
    Interfaces<double, Kokkos::DefaultHostExecutionSpace> faces;
    Cells<double, Kokkos::DefaultHostExecutionSpace> cells;
};

GridInfo build_test_grid() {
    GridInfo grid_info{};
    Vertices<double, Kokkos::DefaultHostExecutionSpace> vertices(16);
    std::vector<Vector3<double>> vertex_pos{
        Vector3<double>(0.0, 0.0, 0.0), Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(2.0, 0.0, 0.0), Vector3<double>(3.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0), Vector3<double>(1.0, 1.0, 0.0),
        Vector3<double>(2.0, 1.0, 0.0), Vector3<double>(3.0, 1.0, 0.0),
        Vector3<double>(0.0, 2.0, 0.0), Vector3<double>(1.0, 2.0, 0.0),
        Vector3<double>(2.0, 2.0, 0.0), Vector3<double>(3.0, 2.0, 0.0),
        Vector3<double>(0.0, 3.0, 0.0), Vector3<double>(1.0, 3.0, 0.0),
        Vector3<double>(2.0, 3.0, 0.0), Vector3<double>(3.0, 3.0, 0.0)};
    for (size_t i = 0; i < 16; i++) {
        vertices.set_vertex_position(i, vertex_pos[i]);
    }

    std::vector<std::vector<size_t>> interface_id_list{
        {0, 1},   {1, 5},  {5, 4},   {4, 0},  {1, 2},   {2, 6},   {6, 5},   {2, 3},
        {3, 7},   {7, 6},  {5, 9},   {9, 8},  {8, 4},   {6, 10},  {10, 9},  {7, 11},
        {11, 10}, {9, 13}, {13, 12}, {12, 8}, {10, 14}, {14, 13}, {11, 15}, {15, 14}};
    std::vector<ElemType> shapes = {
        ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line,
        ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line,
        ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line,
        ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line,
        ElemType::Line, ElemType::Line, ElemType::Line, ElemType::Line,
    };

    Interfaces<double, Kokkos::DefaultHostExecutionSpace> interfaces(interface_id_list,
                                                                     shapes);

    std::vector<std::vector<size_t>> cell_interfaces_list{
        {0, 1, 2, 3},     {4, 5, 6, 1},     {7, 8, 9, 5},
        {2, 10, 11, 12},  {6, 13, 14, 10},  {9, 15, 16, 13},
        {11, 17, 18, 19}, {14, 20, 21, 17}, {16, 22, 23, 20}};

    std::vector<std::vector<size_t>> cell_vertex_ids_raw{
        {0, 1, 5, 4},   {1, 2, 6, 5},   {2, 3, 7, 6},    {4, 5, 9, 8},    {5, 6, 10, 9},
        {6, 7, 11, 10}, {8, 9, 13, 12}, {9, 10, 14, 13}, {10, 11, 15, 14}};

    std::vector<ElemType> cell_shapes{
        ElemType::Quad, ElemType::Quad, ElemType::Quad, ElemType::Quad, ElemType::Quad,
        ElemType::Quad, ElemType::Quad, ElemType::Quad, ElemType::Quad,
    };

    Cells<double, Kokkos::DefaultHostExecutionSpace> cells(
        cell_vertex_ids_raw, cell_interfaces_list, cell_shapes, 9, 0);
    grid_info.vertices = vertices;
    grid_info.faces = interfaces;
    grid_info.cells = cells;
    return grid_info;
}

json build_config() {
    json config{};
    json boundaries{};
    json slip_wall{};
    json inflow{};
    json outflow{};
    slip_wall["ghost_cells"] = true;
    inflow["ghost_cells"] = true;
    outflow["ghost_cells"] = true;
    boundaries["slip_wall_bottom"] = slip_wall;
    boundaries["slip_wall_top"] = slip_wall;
    boundaries["inflow"] = inflow;
    boundaries["outflow"] = outflow;
    config["boundaries"] = boundaries;
    return config;
}

json build_3D_config() {
    json config{};
    json boundaries{};
    json slip_wall{};
    json inflow{};
    json outflow{};
    slip_wall["ghost_cells"] = true;
    inflow["ghost_cells"] = true;
    outflow["ghost_cells"] = true;
    boundaries["bottom"] = slip_wall;
    boundaries["top"] = slip_wall;
    boundaries["west"] = inflow;
    boundaries["east"] = outflow;
    boundaries["north"] = slip_wall;
    boundaries["south"] = slip_wall;
    config["boundaries"] = boundaries;
    return config;
}

TEST_CASE("grid vertices") {
    GridInfo expected = build_test_grid();
    json config = build_config();
    GridBlock<double> block_dev("../../../src/grid/test/grid.su2", config);
    auto block = block_dev.host_mirror();
    block.deep_copy(block_dev);
    auto expected_vertices = expected.vertices;
    CHECK(block.vertices() == expected_vertices);
}

TEST_CASE("grid interfaces") {
    GridInfo expected = build_test_grid();
    json config = build_config();
    GridBlock<double> block_dev("../../../src/grid/test/grid.su2", config);
    auto block = block_dev.host_mirror();
    block.deep_copy(block_dev);
    CHECK(block.interfaces() == expected.faces);
}

TEST_CASE("3D cell volumes") {
    json config = build_3D_config();
    GridBlock<double> block_dev("../../../src/grid/test/cube.su2", config);
    auto block = block_dev.host_mirror();
    block.deep_copy(block_dev);
    size_t n_cells = block.num_cells();
    CHECK(n_cells == 27);
    for (size_t i = 0; i < n_cells; i++) {
        CHECK(block.cells().volume(i) == doctest::Approx(1.0 / n_cells));
    }
}

TEST_CASE("3D interface orientation lengths") {
    json config = build_3D_config();
    GridBlock<double> block_dev("../../../src/grid/test/cube.su2", config);
    auto block = block_dev.host_mirror();
    block.deep_copy(block_dev);
    size_t n_faces = block.num_interfaces();
    for (size_t i = 0; i < n_faces; i++) {
        double t1x = block.interfaces().tan1().x(i);
        double t1y = block.interfaces().tan1().y(i);
        double t1z = block.interfaces().tan1().z(i);
        double t1_length = Ibis::sqrt(t1x * t1x + t1y * t1y + t1z * t1z);
        CHECK(t1_length == doctest::Approx(1.0));

        double t2x = block.interfaces().tan2().x(i);
        double t2y = block.interfaces().tan2().y(i);
        double t2z = block.interfaces().tan2().z(i);
        double t2_length = Ibis::sqrt(t2x * t2x + t2y * t2y + t2z * t2z);
        CHECK(t2_length == doctest::Approx(1.0));

        double nx = block.interfaces().norm().x(i);
        double ny = block.interfaces().norm().y(i);
        double nz = block.interfaces().norm().z(i);
        double n_length = Ibis::sqrt(nx * nx + ny * ny + nz * nz);
        CHECK(n_length == doctest::Approx(1.0));
    }
}

TEST_CASE("grid cell faces") {
    GridInfo expected = build_test_grid();
    json config = build_config();
    GridBlock<double> block_dev("../../../src/grid/test/grid.su2", config);
    auto block = block_dev.host_mirror();
    block.deep_copy(block_dev);
    CHECK(block.cells().num_valid_cells() == expected.cells.num_valid_cells());
    for (size_t i = 0; i < block.num_cells(); i++) {
        for (size_t j = 0; j < block.cells().faces().face_ids(i).size(); j++) {
            CHECK(block.cells().faces().face_ids(i)(j) ==
                  expected.cells.faces().face_ids(i)(j));
        }
    }
}

TEST_CASE("grid cell faces 2") {
    json config = build_config();
    GridBlock<double> block_dev("../../../src/grid/test/grid.su2", config);
    auto block = block_dev.host_mirror();
    block.deep_copy(block_dev);
    std::vector<std::vector<size_t>> face_ids = {
        {0, 1, 2, 3},     {4, 5, 6, 1},     {7, 8, 9, 5},
        {2, 10, 11, 12},  {6, 13, 14, 10},  {9, 15, 16, 13},
        {11, 17, 18, 19}, {14, 20, 21, 17}, {16, 22, 23, 20}};
    CHECK(block.num_cells() == 9);
    for (size_t i = 0; i < block.num_cells(); i++) {
        CellFaces<double>::mirror_type faces = block.cells().faces();
        for (size_t j = 0; j < faces.face_ids(i).size(); j++) {
            CHECK(faces.face_ids(i)(j) == face_ids[i][j]);
        }
    }
}

TEST_CASE("grid cell outsigns") {
    GridInfo expected = build_test_grid();
    json config = build_config();
    GridBlock<double> block_dev("../../../src/grid/test/grid.su2", config);
    auto block = block_dev.host_mirror();
    block.deep_copy(block_dev);
    CHECK(block.cells().num_valid_cells() == expected.cells.num_valid_cells());
    std::vector<std::vector<int>> outsigns = {
        {1, 1, 1, 1},   {1, 1, 1, -1}, {1, 1, 1, -1},  {-1, 1, 1, 1}, {-1, 1, 1, -1},
        {-1, 1, 1, -1}, {-1, 1, 1, 1}, {-1, 1, 1, -1}, {-1, 1, 1, -1}};
    for (size_t i = 0; i < block.cells().num_valid_cells(); i++) {
        for (size_t j = 0; j < block.cells().faces().outsigns(i).size(); j++) {
            CHECK(block.cells().faces().outsigns(i)(j) == outsigns[i][j]);
        }
    }
}

TEST_CASE("cell neighbours") {
    json config = build_config();
    GridBlock<double> block_dev("../../../src/grid/test/grid.su2", config);
    auto block_host = block_dev.host_mirror();
    block_host.deep_copy(block_dev);

    CHECK(block_host.cells().neighbour_cells(0, 1) == 1);
    CHECK(block_host.cells().neighbour_cells(0, 2) == 3);
    CHECK(block_host.cells().neighbour_cells(4, 0) == 1);
    CHECK(block_host.cells().neighbour_cells(8, 3) == 7);
}

TEST_CASE("ghost cell centres") {
    json config = build_config();
    GridBlock<double> block_dev("../../../src/grid/test/grid.su2", config);
    auto block_host = block_dev.host_mirror();
    block_host.deep_copy(block_dev);

    // test the inflow cells
    auto inflow_ghost_cells = block_host.ghost_cells("inflow");
    size_t ghost_cell = inflow_ghost_cells(0);
    CHECK(block_host.cells().centroids().x(ghost_cell) == -0.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 0.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = inflow_ghost_cells(1);
    CHECK(block_host.cells().centroids().x(ghost_cell) == -0.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 1.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = inflow_ghost_cells(2);
    CHECK(block_host.cells().centroids().x(ghost_cell) == -0.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 2.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    // test slip_wall_bottom
    auto bottom_ghost_cells = block_host.ghost_cells("slip_wall_bottom");
    ghost_cell = bottom_ghost_cells(0);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 0.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == -0.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = bottom_ghost_cells(1);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 1.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == -0.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = bottom_ghost_cells(2);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 2.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == -0.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    // test outflow
    auto outflow_ghost_cells = block_host.ghost_cells("outflow");
    ghost_cell = outflow_ghost_cells(0);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 3.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 0.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = outflow_ghost_cells(1);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 3.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 1.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = outflow_ghost_cells(2);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 3.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 2.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    // test slip_wall_top
    auto top_ghost_cells = block_host.ghost_cells("slip_wall_top");
    ghost_cell = top_ghost_cells(0);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 0.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 3.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = top_ghost_cells(1);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 1.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 3.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);

    ghost_cell = top_ghost_cells(2);
    CHECK(block_host.cells().centroids().x(ghost_cell) == 2.5);
    CHECK(block_host.cells().centroids().y(ghost_cell) == 3.5);
    CHECK(block_host.cells().centroids().z(ghost_cell) == 0.0);
}
