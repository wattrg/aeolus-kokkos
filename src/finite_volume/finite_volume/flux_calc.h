#ifndef FLUX_H
#define FLUX_H

#include <finite_volume/conserved_quantities.h>
#include <gas/flow_state.h>
#include <gas/gas_model.h>
#include <grid/interface.h>

#include <nlohmann/json.hpp>

// enum class FluxCalculator {
//     Hanel,
//     Ausmdv,
//     Ldfss,
// };

// FluxCalculator flux_calculator_from_string(std::string name);
// std::string string_from_flux_calculator(FluxCalculator flux_calc);

using json = nlohmann::json;

template <typename T>
class FluxCalculator {
public:
    FluxCalculator(std::string name) : name_(name) {}

    virtual ~FluxCalculator() {}

    virtual void compute_flux(const FlowStates<T>& left, const FlowStates<T>& right,
                              ConservedQuantities<T>& flux, IdealGas<T>& gm,
                              bool three_d) = 0;

    std::string name() { return name_; };

protected:
    std::string name_;
};

template <typename T>
class Hanel : public FluxCalculator<T> {
public:
    Hanel() : FluxCalculator<T>("hanel") {}

    void compute_flux(const FlowStates<T>& left, const FlowStates<T>& right,
                      ConservedQuantities<T>& flux, IdealGas<T>& gm, bool three_d);
};

template <typename T>
class Ausmdv : public FluxCalculator<T> {
public:
    Ausmdv() : FluxCalculator<T>("ausmdv") {}

    void compute_flux(const FlowStates<T>& left, const FlowStates<T>& right,
                      ConservedQuantities<T>& flux, IdealGas<T>& gm, bool three_d);
};

template <typename T>
class Ldfss2 : public FluxCalculator<T> {
public:
    Ldfss2() : FluxCalculator<T>("ldfss2") {}

    Ldfss2(json config) : FluxCalculator<T>("ldfss2") { delta_ = config.at("delta"); }

    void compute_flux(const FlowStates<T>& left, const FlowStates<T>& right,
                      ConservedQuantities<T>& flux, IdealGas<T>& gm, bool three_d);

private:
    Ibis::real delta_;
};

template <typename T>
std::unique_ptr<FluxCalculator<T>> make_flux_calculator(json config);

#endif
