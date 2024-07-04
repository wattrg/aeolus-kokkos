#include <finite_volume/flux_calc.h>
#include <gas/gas_model.h>
#include <gas/gas_state.h>
#include <pybind11/pybind11.h>
#include <util/vector3.h>

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <Kokkos_Core.hpp>

using ideal_gas = IdealGas<Ibis::real>;

PYBIND11_MODULE(python_api, m) {
    m.doc() = "ibis python module";

    pybind11::class_<GasState<Ibis::real>>(m, "GasState")
        .def(pybind11::init())
        .def_readwrite("p", &GasState<Ibis::real>::pressure, "pressure (Pa)")
        .def_readwrite("T", &GasState<Ibis::real>::temp, "temperature (K)")
        .def_readwrite("rho", &GasState<Ibis::real>::rho, "density (kg/m^3)")
        .def_readwrite("energy", &GasState<Ibis::real>::energy, "energy (J/kg)");

    pybind11::class_<ideal_gas>(m, "PyIdealGas")
        .def(pybind11::init<Ibis::real>())
        .def("R", &ideal_gas::R)
        .def("Cv", &ideal_gas::Cv)
        .def("Cp", &ideal_gas::Cp)
        .def("gamma", &ideal_gas::gamma)
        .def("speed_of_sound",
             static_cast<Ibis::real (ideal_gas::*)(const GasState<Ibis::real>&) const>(
                 &ideal_gas::speed_of_sound))
        .def("update_thermo_from_pT",
             static_cast<void (ideal_gas::*)(GasState<Ibis::real>&) const>(
                 &ideal_gas::update_thermo_from_pT))
        .def("update_thermo_from_rhoT",
             static_cast<void (ideal_gas::*)(GasState<Ibis::real>&) const>(
                 &ideal_gas::update_thermo_from_rhoT))
        .def("update_thermo_from_rhop",
             static_cast<void (ideal_gas::*)(GasState<Ibis::real>&) const>(
                 &ideal_gas::update_thermo_from_rhop));

    pybind11::class_<Vector3<Ibis::real>>(m, "Vector3")
        .def(pybind11::init<>())
        .def(pybind11::init<Ibis::real>())
        .def(pybind11::init<Ibis::real, Ibis::real>())
        .def(pybind11::init<Ibis::real, Ibis::real, Ibis::real>())
        .def_readwrite("x", &Vector3<Ibis::real>::x, "x")
        .def_readwrite("y", &Vector3<Ibis::real>::y, "y")
        .def_readwrite("z", &Vector3<Ibis::real>::z, "z");

    // flux calculators
    pybind11::class_<FluxCalculator<Ibis::real>>(m, "FluxCalculator");

    pybind11::class_<Hanel<Ibis::real>>(m, "PyHanel")
        .def(pybind11::init<>())
        .def("name", &Hanel<Ibis::real>::name);

    pybind11::class_<Ausmdv<Ibis::real>>(m, "PyAusmdv")
        .def(pybind11::init<>())
        .def("name", &Ausmdv<Ibis::real>::name);

    pybind11::class_<Ldfss2<Ibis::real>>(m, "PyLdfss2")
        .def(pybind11::init<>())
        .def("name", &Ldfss2<Ibis::real>::name);
}
