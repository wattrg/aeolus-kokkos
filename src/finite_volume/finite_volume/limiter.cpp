#include <finite_volume/limiter.h>
#include <spdlog/spdlog.h>

#include <stdexcept>

template <typename T>
std::unique_ptr<Limiter<T>> make_limiter(json config) {
    std::string limiter_type = config.at("type");
    if (limiter_type == "barth_jespersen"){
        return std::unique_ptr<Limiter<T>>(new BarthJespersen<T>(config.at("epsilon")));
    } else if (limiter_type == "unlimited"){
        return std::unique_ptr<Limiter<T>>(new Unlimited<T>());
    } else {
        spdlog::error("Unknown limiter {}", limiter_type);
        throw new std::runtime_error("Unknown limiter");
    }
}
template std::unique_ptr<Limiter<double>> make_limiter<double>(json);


template <typename T>
void BarthJespersen<T>::calculate_limiters(const Ibis::SubArray2D<T> values, Field<T>& limits,
                                           const Cells<T>& cells, const Interfaces<T>& faces,
                                           Vector3s<T>& grad) {
    double epsilon = epsilon_;
    Kokkos::parallel_for(
        "Limiter::barth_jesperson", cells.num_valid_cells(),
        KOKKOS_LAMBDA(const int cell_i) {
            T Ui = values(cell_i);
            T U_min = Ui;
            T U_max = Ui;
            for (size_t j = 0; j < cells.neighbour_cells(cell_i).size(); j++) {
                int neighbour_cell = cells.neighbour_cells(cell_i, j);
                U_min = Kokkos::min(U_min, values(neighbour_cell));
                U_max = Kokkos::max(U_max, values(neighbour_cell));
            }

            T phi = 1.0;
            auto face_ids = cells.faces().face_ids(cell_i);
            for (size_t j = 0; j < face_ids.size(); j++) {
                int i_face = face_ids(j);
                T dx = faces.centre().x(i_face);
                T dy = faces.centre().y(i_face);
                T dz = faces.centre().z(i_face);
                T delta_2 =
                    grad.x(cell_i) * dx + grad.y(cell_i) * dy + grad.z(cell_i) * dz;
                int sign_delta_2 = (delta_2 > 0) - (delta_2 < 0);
                delta_2 = sign_delta_2 * (Kokkos::abs(delta_2) + epsilon);
                if (sign_delta_2 > 0) {
                    phi = Kokkos::min(phi, (U_max - Ui) / delta_2);
                } else if (sign_delta_2 < 0) {
                    phi = Kokkos::min(phi, (U_min - Ui) / delta_2);
                }
            }
            limits(cell_i) = phi;
        });
}

template class BarthJespersen<double>;

template <typename T>
void Unlimited<T>::calculate_limiters(const Ibis::SubArray2D<T> values, Field<T>& limits,
                                           const Cells<T>& cells, const Interfaces<T>& faces,
                                           Vector3s<T>& grad) {
    (void) values;
    (void) limits;
    (void) cells;
    (void) faces;
    (void) grad;
}
template class Unlimited<double>;
