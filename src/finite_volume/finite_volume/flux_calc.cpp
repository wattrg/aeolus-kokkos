#include <doctest/doctest.h>
#include <finite_volume/flux_calc.h>
#include <spdlog/spdlog.h>
#include <util/numeric_types.h>

#include <stdexcept>

template <typename T>
std::unique_ptr<FluxCalculator<T>> make_flux_calculator(json config) {
    std::string type = config.at("type");
    if (type == "hanel") {
        return std::unique_ptr<FluxCalculator<T>>(new Hanel<T>());
    } else if (type == "ausmdv") {
        return std::unique_ptr<FluxCalculator<T>>(new Ausmdv<T>());
    } else if (type == "ldfss2") {
        return std::unique_ptr<FluxCalculator<T>>(new Ldfss2<T>(config));
    } else {
        spdlog::error("Unknown flux calculator {}", type);
        throw std::runtime_error("Unknown flux calculator");
    }
}

template std::unique_ptr<FluxCalculator<Ibis::real>> make_flux_calculator(json);
template std::unique_ptr<FluxCalculator<Ibis::dual>> make_flux_calculator(json);
