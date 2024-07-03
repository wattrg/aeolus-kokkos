#include <finite_volume/limiter.h>
#include <spdlog/spdlog.h>
#include <util/numeric_types.h>

#include <stdexcept>

template <typename T>
std::unique_ptr<Limiter<T>> make_limiter(json config) {
    std::string limiter_type = config.at("type");
    if (limiter_type == "barth_jespersen") {
        Ibis::real epsilon = config.at("epsilon");
        return std::unique_ptr<Limiter<T>>(new BarthJespersen<T>(epsilon));
    } else if (limiter_type == "unlimited") {
        return std::unique_ptr<Limiter<T>>(new Unlimited<T>());
    } else {
        spdlog::error("Unknown limiter {}", limiter_type);
        throw new std::runtime_error("Unknown limiter");
    }
}
template std::unique_ptr<Limiter<Ibis::real>> make_limiter<Ibis::real>(json);
template std::unique_ptr<Limiter<Ibis::dual>> make_limiter<Ibis::dual>(json);

template <typename T>
void BarthJespersen<T>::calculate_limiters(const Ibis::SubArray2D<T> values,
                                           Field<T>& limits, const Cells<T>& cells,
                                           const Interfaces<T>& faces,
                                           Vector3s<T>& grad) {
    Ibis::real epsilon = epsilon_;
    Kokkos::parallel_for(
        "Limiter::barth_jesperson", cells.num_valid_cells(),
        KOKKOS_LAMBDA(const size_t cell_i) {
            T Ui = values(cell_i);
            T U_min = Ui;
            T U_max = Ui;
            for (size_t j = 0; j < cells.neighbour_cells(cell_i).size(); j++) {
                size_t neighbour_cell = cells.neighbour_cells(cell_i, j);
                U_min = Ibis::min(U_min, values(neighbour_cell));
                U_max = Ibis::max(U_max, values(neighbour_cell));
            }

            T phi = 1.0;
            T x = cells.centroids().x(cell_i);
            T y = cells.centroids().y(cell_i);
            T z = cells.centroids().z(cell_i);
            auto face_ids = cells.faces().face_ids(cell_i);
            for (size_t j = 0; j < face_ids.size(); j++) {
                int i_face = face_ids(j);
                T dx = faces.centre().x(i_face) - x;
                T dy = faces.centre().y(i_face) - y;
                T dz = faces.centre().z(i_face) - z;
                T delta_2 =
                    grad.x(cell_i) * dx + grad.y(cell_i) * dy + grad.z(cell_i) * dz;
                int sign_delta_2 = (delta_2 > 0) - (delta_2 < 0);
                delta_2 = sign_delta_2 * (Ibis::abs(delta_2) + epsilon);
                if (sign_delta_2 > 0) {
                    phi = Ibis::min(phi, (U_max - Ui) / delta_2);
                } else if (sign_delta_2 < 0) {
                    phi = Ibis::min(phi, (U_min - Ui) / delta_2);
                }
            }
            limits(cell_i) = phi;
        });
}
template class BarthJespersen<Ibis::real>;
template class BarthJespersen<Ibis::dual>;

// template <typename T>
// void Venkat<T>::calculate_limiters(const Ibis::SubArray2D<T> values, Field<T>& limits,
//                                    const Cells<T>& cells, const Interfaces<T>& faces,
//                                    Vector3s<T>& grad) {
//     double K = K_;
//     Kokkos::parallel_for(
//         "Limiter::venkat", cells.num_valid_cells(),
//         KOKKOS_LAMBDA(const size_t cell_i){
//             T Ui = values(cell_i);
//             T U_min = Ui;
//             T U_max = Ui;
//             for (size_t j = 0; j < cells.neighbour_cells(cell_i).size(); j++) {
//                 size_t neighbour_cell = cells.neighbour_cells(cell_i, j);
//                 U_min = Ibis::min(U_min, values(neighbour_cell));
//                 U_max = Ibis::max(U_max, values(neighbour_cell));
//             }
//             T delta_1_max = U_max - Ui;
//             T delta_1_min = U_min - Ui;

//             T phi = 1.0;
//             T x = cells.centroids().x(cell_i);
//             T y = cells.centroids().y(cell_i);
//             T z = cells.centroids().z(cell_i);
//             T eps2 = K * K * K * cells.volume(cell_i);
//             auto face_ids = cells.faces().face_ids(cell_i);
//             for (size_t j = 0; j < face_ids.size(); j++){
//                 int i_face = face_ids(j);
//                 T dx = faces.centre().x(i_face) - x;
//                 T dy = faces.centre().y(i_face) - y;
//                 T dz = faces.centre().z(i_face) - z;
//                 T delta_2 =
//                     grad.x(cell_i) * dx + grad.y(cell_i) * dy + grad.z(cell_i) * dz;
//                 if (delta_2 > 0) {
//                     // nothing for the moment
//                 }
//                 else if (delta_2 > 0) {
//                     // nothing for the moment
//                 }
//             }
//             limits(cell_i) = phi;
//         });
// }
// template class Venkat<double>;

template <typename T>
void Unlimited<T>::calculate_limiters(const Ibis::SubArray2D<T> values, Field<T>& limits,
                                      const Cells<T>& cells, const Interfaces<T>& faces,
                                      Vector3s<T>& grad) {
    (void)values;
    (void)limits;
    (void)cells;
    (void)faces;
    (void)grad;
}
template class Unlimited<Ibis::real>;
template class Unlimited<Ibis::dual>;
