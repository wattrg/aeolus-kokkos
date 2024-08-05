import json
import os
import shutil
from enum import Enum
import struct

from ibis_py_utils import (
    ConstantPrandtlNumber,
    SutherlandViscosity,
    TransportPropertyModel,
    read_defaults,
    FlowState,
    GasModel
)

from python_api import (
    PyAusmdv,
    PyHanel,
    PyLdfss2,
    GasState,
    PyIdealGas
)

validation_errors = []


class ValidationException(Exception):
    pass


class Solver(Enum):
    RungeKutta = "runge_kutta"
    SteadyState = "steady_state"


def string_to_solver(string):
    if string == Solver.RungeKutta.value:
        return Solver.RungeKutta
    elif string == Solver.SteadyState.value:
        return Solver.SteadyState
    validation_errors.append(ValidationException(f"Unknown solver {string}"))


class Limiter:
    def _read_defaults(self):
        json_data = read_defaults(DEFAULTS_DIRECTORY, self._defaults_file)
        for key in self._json_values:
            setattr(self, key, json_data[key])


class Unlimited(Limiter):
    _defaults_file = "unlimited.json"
    _json_values = []

    def __init__(self):
        self._read_defaults()
        self._name = "unlimited"

    def as_dict(self):
        return {"type": self._name}


class BarthJespersen(Limiter):
    _defaults_file = "barth_jespersen.json"
    _json_values = ["epsilon"]
    __slots__ = _json_values

    def __init__(self, **kwargs):
        self._read_defaults()
        self._name = "barth_jespersen"

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def as_dict(self):
        dictionary = {"type": self._name, }
        for key in self._json_values:
            dictionary[key] = getattr(self, key)
        return dictionary


def string_to_limiter(string):
    if string == "barth_jespersen":
        return BarthJespersen()
    if string == "unlimited":
        return Unlimited()
    validation_errors.append(ValidationException(f"Unknown limiter {string}"))


class FluxCalculator:
    def _read_defaults(self):
        json_data = read_defaults(DEFAULTS_DIRECTORY, self._defaults_file)
        for key in self._json_values:
            setattr(self, key, json_data[key])


class Hanel(FluxCalculator):
    _defaults_file = "hanel.json"
    _json_values = []

    def __init__(self):
        self._read_defaults()
        self._flux_calc = PyHanel()

    def as_dict(self):
        return {"type": self._flux_calc.name()}


class Ausmdv(FluxCalculator):
    _defaults_file = "ausmdv.json"
    _json_values = []

    def __init__(self):
        self._read_defaults()
        self._flux_calc = PyAusmdv()

    def as_dict(self):
        self._read_defaults()
        return {"type": self._flux_calc.name()}


class Ldfss2(FluxCalculator):
    _defaults_file = "ldfss2.json"
    _json_values = ["delta"]

    def __init__(self, delta=None):
        self._read_defaults()
        self._flux_calc = PyLdfss2()

        if delta:
            self.delta = delta

    def as_dict(self):
        return {"type": self._flux_calc.name(), "delta": self.delta}


def string_to_flux_calc(name):
    if name == "hanel":
        return Hanel()
    elif name == "ausmdv":
        return Ausmdv()
    elif name == "ldfss2":
        return Ldfss2()
    else:
        validation_errors.append(
            ValidationException(f"Unknown flux calculator {name}")
        )


class ThermoInterp(Enum):
    RhoP = "rho_p"
    RhoT = "rho_T"
    RhoU = "rho_u"
    PT = "p_T"


def string_to_thermo_interp(name):
    if name == ThermoInterp.RhoP.value:
        return ThermoInterp.RhoP
    if name == ThermoInterp.RhoT.value:
        return ThermoInterp.RhoT
    if name == ThermoInterp.RhoU.value:
        return ThermoInterp.RhoU
    if name == ThermoInterp.PT.value:
        return ThermoInterp.PT


def ensure_custom_type(value, conversion_func):
    if type(value) is str:
        return conversion_func(value)
    return value


class ConvectiveFlux:
    _json_values = ["flux_calculator", "reconstruction_order", "limiter",
                    "thermo_interpolator"]
    _custom_types = {
        "flux_calculator": string_to_flux_calc,
        "limiter": string_to_limiter,
        "thermo_interpolator": string_to_thermo_interp,
    }
    __slots__ = _json_values
    _defaults_file = "convective_flux.json"

    def __init__(self, **kwargs):
        json_data = read_defaults(DEFAULTS_DIRECTORY,
                                  self._defaults_file)
        for key in self._json_values:
            if key in self._custom_types:
                setattr(self, key, ensure_custom_type(json_data[key],
                                                      self._custom_types[key]))
            else:
                setattr(self, key, json_data[key])

        for key in kwargs:
            if key in self._custom_types:
                setattr(self, key, ensure_custom_type(kwargs[key],
                                                      self._custom_types[key]))
            else:
                setattr(self, key, kwargs[key])

    def validate(self):
        if self.reconstruction_order not in (1, 2):
            validation_errors.append(
                ValidationException(
                    f"reconstruction order {self.reconstruction_order}"
                    " not supported"
                )
            )

    def as_dict(self):
        dictionary = {}
        for key in self._json_values:
            if key == "flux_calculator":
                dictionary[key] = self.flux_calculator.as_dict()
            elif key == "limiter":
                dictionary[key] = self.limiter.as_dict()
            elif key == "thermo_interpolator":
                interp = self.thermo_interpolator
                if type(interp) is str:
                    self.thermo_interpolator = string_to_thermo_interp(interp)
                dictionary[key] = self.thermo_interpolator.value
            else:
                dictionary[key] = getattr(self, key)
        return dictionary


class ViscousFlux:
    _json_values = ["enabled", "signal_factor"]
    __slots__ = _json_values
    _defaults_file = "viscous_flux.json"

    def __init__(self, **kwargs):
        json_data = read_defaults(DEFAULTS_DIRECTORY,
                                  self._defaults_file)
        for key in self._json_values:
            setattr(self, key, json_data[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def validate(self):
        if self.signal_factor < 0:
            validation_errors.append(ValidationException(
                f"Invalid signal_factor {self.signal_factor}")
            )

    def as_dict(self):
        dictionary = {}
        for key in self._json_values:
            dictionary[key] = getattr(self, key)
        return dictionary


class Block:
    def __init__(self, file_name, initial_condition, boundaries):
        self._initial_condition = initial_condition
        self._block = file_name
        self.number_cells = 0
        self.number_vertices = 0
        self.boundaries = boundaries
        self._read_file(file_name)

    def _read_file(self, file_name):
        extension = file_name.split(".")[-1]
        if extension == "su2":
            self._read_su2_grid(file_name)
        else:
            raise Exception(f"Unknown grid format: {extension}")

    def _read_su2_grid(self, file_name):
        with open(file_name, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("NDIME"):
                    self.dim = int(line.split("=")[-1].strip())
                elif line.startswith("NPOIN"):
                    self.number_vertices = int(line.split("=")[-1].strip())
                elif line.startswith("NELEM"):
                    self.number_cells = int(line.split("=")[-1].strip())
                elif line.startswith("NMARK"):
                    self.number_boundaries = int(line.split("=")[-1].strip())

    def validate(self):
        if not self._block:
            validation_errors.append(
                ValidationException("No grid blocks specified")
            )

    def _number(self, number, binary):
        if binary:
            return struct.pack("d", number)
        return f"{number:.16e}\n"

    def write(self, grid_directory, flow_directory, binary):
        if not os.path.exists(grid_directory):
            os.mkdir(grid_directory)
        if not os.path.exists(flow_directory):
            os.mkdir(flow_directory)
        ic_path = f"{flow_directory}/0000"
        if not os.path.exists(ic_path):
            os.mkdir(ic_path)

        # write the grid
        shutil.copy(self._block, f"{grid_directory}/block_{0:04}.su2")

        # write the initial condition
        format = "wb" if binary else "w"
        ic_directory = f"{flow_directory}/{0:04}"
        temp = open(f"{ic_directory}/T", format)
        pressure = open(f"{ic_directory}/p", format)
        vx = open(f"{ic_directory}/vx", format)
        vy = open(f"{ic_directory}/vy", format)
        if self.dim == 3:
            vz = open(f"{ic_directory}/vz", format)
        meta_data = open(f"{ic_directory}/meta_data.json", "w")
        times = open(f"{flow_directory}/flows", "w")

        if type(self._initial_condition) is FlowState:
            for _ in range(self.number_cells):
                temp.write(self._number(self._initial_condition.gas.T, binary))
                pressure.write(self._number(self._initial_condition.gas.p,
                                            binary))
                vx.write(self._number(self._initial_condition.vel.x, binary))
                vy.write(self._number(self._initial_condition.vel.y, binary))
                if self.dim == 3:
                    vz.write(self._number(self._initial_condition.vel.z,
                                          binary))
        json.dump({"time": 0.0}, meta_data, indent=4)
        times.write("0000\n")

        temp.close()
        pressure.close()
        vx.close()
        vy.close()
        if self.dim == 3:
            vz.close()
        meta_data.close()
        times.close()

    def as_dict(self):
        dictionary = {"boundaries": {}}
        dictionary["dimensions"] = self.dim
        for key in self.boundaries:
            dictionary["boundaries"][key] = self.boundaries[key].as_dict()
        return dictionary


class BoundaryCondition:
    def __init__(self, pre_reconstruction, pre_viscous_grad, ghost_cells=True):
        self._pre_reconstruction = pre_reconstruction
        self._pre_viscous_grad = pre_viscous_grad
        self.ghost_cells = ghost_cells

    def as_dict(self):
        dictionary = {}
        pre_reco_dict = []
        for pre_reco in self._pre_reconstruction:
            pre_reco_dict.append(pre_reco.as_dict())
        dictionary["pre_reconstruction"] = pre_reco_dict

        pre_viscous_grad_dict = []
        for pre_viscous_grad in self._pre_viscous_grad:
            pre_viscous_grad_dict.append(pre_viscous_grad.as_dict())
        dictionary["pre_viscous_grad"] = pre_viscous_grad_dict

        dictionary["ghost_cells"] = self.ghost_cells
        return dictionary


class _FlowStateCopy:
    def __init__(self, flow_state):
        self.flow_state = flow_state

    def as_dict(self):
        return {
            "type": "flow_state_copy",
            "flow_state": self.flow_state.as_dict()
        }


class _BoundaryLayerProfile:
    def __init__(self, height, vel_profile, temp_profile, pressure):
        self.height = height
        self.vel_profile = vel_profile
        self.temp_profile = temp_profile
        self.pressure = pressure

    def as_dict(self):
        return {
            "type": "boundary_layer_profile",
            "profile": {
                "height": self.height,
                "v": self.vel_profile,
                "T": self.temp_profile,
                "p": self.pressure,
            }
        }


class _InternalCopy:
    def as_dict(self):
        return {"type": "internal_copy"}


class _InternalCopyReflectNormal:
    def as_dict(self):
        return {"type": "internal_copy_reflect_normal"}


class _InternalVelCopyReflect:
    def as_dict(self):
        return {"type": "internal_vel_copy_reflect"}


class _FixTemperature:
    def __init__(self, temperature):
        self._temperature = temperature

    def as_dict(self):
        return {
            "type": "fix_temperature",
            "temperature": self._temperature
        }


class _SubsonicInflow:
    def __init__(self, flow_state):
        self._flow_state = flow_state

    def as_dict(self):
        return {
            "type": "subsonic_inflow",
            "flow_state": self._flow_state.as_dict()
        }


class _SubsonicOutflow:
    def __init__(self, pressure):
        self._pressure = pressure

    def as_dict(self):
        return {
            "type": "subsonic_outflow",
            "pressure": self._pressure
        }


def supersonic_inflow(inflow):
    return BoundaryCondition(
        pre_reconstruction=[_FlowStateCopy(inflow)],
        pre_viscous_grad=[]
    )


def boundary_layer_inflow(height, velocity_profile,
                          temperature_profile, pressure):
    return BoundaryCondition(
        pre_reconstruction=[_BoundaryLayerProfile(height, velocity_profile,
                                                  temperature_profile,
                                                  pressure)],
        pre_viscous_grad=[]
    )


def supersonic_outflow():
    return BoundaryCondition(
        pre_reconstruction=[_InternalCopy()],
        pre_viscous_grad=[]
    )


def slip_wall():
    return BoundaryCondition(
        pre_reconstruction=[_InternalCopyReflectNormal()],
        pre_viscous_grad=[]
    )


def adiabatic_no_slip_wall():
    return BoundaryCondition(
        pre_reconstruction=[_InternalCopyReflectNormal()],
        pre_viscous_grad=[_InternalVelCopyReflect()]
    )


def fixed_temperature_no_slip_wall(temperature):
    return BoundaryCondition(
        pre_reconstruction=[_InternalCopyReflectNormal()],
        pre_viscous_grad=[_InternalVelCopyReflect(),
                          _FixTemperature(temperature)]
    )


def subsonic_inflow(flow_state):
    return BoundaryCondition(
        pre_reconstruction=[_SubsonicInflow(flow_state)],
        pre_viscous_grad=[]
    )


def subsonic_outflow(pressure):
    return BoundaryCondition(
        pre_reconstruction=[_SubsonicOutflow(pressure)],
        pre_viscous_grad=[]
    )


class CflSchedule:
    def as_dict(self):
        pass


class ConstantCfl(CflSchedule):
    _type = "constant"

    def __init__(self, cfl):
        self._cfl = cfl

    def as_dict(self):
        return {"type": self._type, "value": self._cfl}


class LinearInterpolateCfl(CflSchedule):
    _type = "linear_interpolate"

    def __init__(self, cfls):
        self._times = []
        self._cfls = []
        for time, cfl in cfls:
            self._times.append(time)
            self._cfls.append(cfl)

    def as_dict(self):
        return {
            "type": self._type,
            "times": self._times,
            "cfls": self._cfls
        }


def make_cfl_schedule(config):
    if type(config) is float:
        return ConstantCfl(config)

    cfl_type = config["type"]
    if cfl_type == "constant":
        return ConstantCfl(config["value"])
    elif cfl_type == "linear_interpolate":
        cfls = []
        for time, cfl in zip(config["times"], config["cfls"]):
            cfls.append((time, cfl))
        return LinearInterpolateCfl(cfls)
    else:
        validation_errors.append(
            ValidationException(f"Unkown cfl schedule {cfl_type}")
        )


class ButcherTableau:
    def __init__(self, a, b, c):
        self._a = a
        self._b = b
        self._c = c

    def n_stages(self):
        return len(self._b)

    def as_dict(self):
        return {"a": self._a, "b": self._b, "c": self._c}


def butcher_tableau(method):
    if method == "euler":
        return ButcherTableau([[]], [1.0], [])
    elif method == "midpoint":
        return ButcherTableau([[0.5]], [0, 1], [0.5])
    elif method == "rk3":
        return ButcherTableau([[1/2], [-1, 2]], [1/6, 2/3, 1/6], [0.5, 1.0])
    elif method == "ssp-rk3":
        return ButcherTableau([[1], [0.25, 0.25]], [1/6, 1/6, 2/3], [1, 0.5])
    elif method == "rk4":
        return ButcherTableau([[0.5], [0, 0.5], [0, 0, 1]],
                              [1/6, 1/3, 1/3, 1/6],
                              [0.5, 0.5, 1.0])
    else:
        raise ValidationException(f"Unknown method {method}")


class RungeKutta:
    _json_values = ["cfl", "max_time", "max_step", "print_frequency",
                    "plot_frequency", "plot_every_n_steps", "dt_init",
                    "method", "butcher_tableau",
                    "residual_frequency", "residuals_every_n_steps"]
    _defaults_file = "runge_kutta.json"
    _name = Solver.RungeKutta.value
    __slots__ = _json_values

    def __init__(self, **kwargs):
        json_data = read_defaults(DEFAULTS_DIRECTORY,
                                  self._defaults_file)
        if "butcher_tableau" in kwargs and "method" in kwargs:
            validation_errors.append(
                ValidationException("butcher_tableau and method provided")
            )

        for key in json_data:
            # we have a default method, from which the butcher_tableau will
            # be generated, but we don't explicitly have a default Butcher
            # tableau, so we'll just skip this key
            if key == "butcher_tableau":
                continue
            setattr(self, key, json_data[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def as_dict(self):
        dictionary = {"name": self._name}

        for key in self._json_values:
            if key == "cfl" and type(self.cfl) is not CflSchedule:
                self.cfl = make_cfl_schedule(self.cfl).as_dict()
            if key == "method" and not hasattr(self, "butcher_tableau"):
                self.butcher_tableau = butcher_tableau(self.method).as_dict()
                continue
            dictionary[key] = getattr(self, key)
        return dictionary

    def validate(self):
        return


class Gmres:
    _json_values = ["max_iters", "tol"]
    __slots__ = _json_values
    _defaults_file = "gmres.json"

    def __init__(self, **kwargs):
        json_data = read_defaults(DEFAULTS_DIRECTORY, self._defaults_file)

        for key in json_data:
            setattr(self, key, json_data[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def as_dict(self):
        dictionary = {}
        for key in self._json_values:
            dictionary[key] = getattr(self, key)
        return dictionary

    def validate(self):
        return


class SteadyState:
    _json_values = ["cfl", "max_steps", "print_frequency", "plot_frequency",
                    "diagnostics_frequency", "tolerance"]
    _defaults_file = "steady_state.json"
    _name = Solver.SteadyState.value
    __slots__ = _json_values + ["linear_solver", "cfl"]

    def __init__(self, **kwargs):
        json_data = read_defaults(DEFAULTS_DIRECTORY, self._defaults_file)

        for key in json_data:
            setattr(self, key, json_data[key])

        self.linear_solver = Gmres()

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def as_dict(self):
        dictionary = {"name": self._name}
        for key in self._json_values:
            if key == "cfl" and type(self.cfl) is not CflSchedule:
                self.cfl = make_cfl_schedule(self.cfl).as_dict()
            dictionary[key] = getattr(self, key)
        dictionary["linear_solver"] = self.linear_solver.as_dict()
        return dictionary

    def validate(self):
        return


def make_default_solver():
    default_solver_name = read_defaults(DEFAULTS_DIRECTORY,
                                        "config.json")["solver"]
    default_solver = string_to_solver(default_solver_name)
    if default_solver == Solver.RungeKutta:
        return RungeKutta()
    elif default_solver == Solver.SteadyState:
        return SteadyState()
    validation_errors.append(
        ValidationException(f"Unknown default solver {default_solver_name}")
    )


def load_species_data(species):
    with open(f"{SHARE_DIRECTORY}/species_database/{species}.json") as f:
        species_data = json.load(f)
    return species_data


class IdealGas(GasModel):
    _type = "ideal_gas"

    def __init__(self, **args):
        if "R" in args:
            self._gas_model = PyIdealGas(args["R"])
        elif "species" in args:
            species = args["species"]
            if type(species) is str:
                self._species = species
            if type(species) is list or type(species) is tuple:
                if len(species) != 1:
                    raise Exception("Multiple species not supported yet")
                self._species = species[0]

            species_data = load_species_data(self._species)
            R = species_data["thermo"]["R"]
            self._gas_model = PyIdealGas(R)

    def as_dict(self):
        return {
            "type": self._type,
            "R": self._gas_model.R(),
            "Cv": self._gas_model.Cv(),
            "Cp": self._gas_model.Cp(),
            "gamma": self._gas_model.gamma()
        }


def default_gas_model():
    defaults = read_defaults(DEFAULTS_DIRECTORY, "config.json")
    default_gas_model = defaults["gas_model"]
    if default_gas_model["model"] == "ideal_gas":
        return IdealGas(species=default_gas_model["species"])


def build_viscosity_model(gas_model):
    gas_model_type = gas_model.type()
    with open(f"{DEFAULTS_DIRECTORY}/transport_properties.json") as f:
        models = json.load(f)
    viscosity_model = models["viscosity"][gas_model_type]
    species_data = load_species_data(gas_model.species())["transport"]
    if viscosity_model == "sutherland":
        species_data = species_data["sutherland"]
        mu_0 = species_data["mu_0"]
        T_0 = species_data["T_0"]
        T_s = species_data["T_s"]
        return SutherlandViscosity(mu_0, T_0, T_s)
    else:
        validation_errors.append(
            ValidationException(f"Unknown default viscosity "
                                f"model {viscosity_model}")
        )


def build_thermal_conductivity_model(gas_model):
    gas_model_type = gas_model.type()
    with open(f"{DEFAULTS_DIRECTORY}/transport_properties.json") as f:
        models = json.load(f)
    thermal_conductivity = models["thermal_conductivity"][gas_model_type]
    species_data = load_species_data(gas_model.species())["transport"]
    if thermal_conductivity == "constant_prandtl":
        Pr = species_data["prandtl"]
        return ConstantPrandtlNumber(Pr)
    else:
        validation_errors.append(
            ValidationException("Unknown default thermal conductivity "
                                f"model {thermal_conductivity}")
        )


def build_transport_property_model(gas_model):
    viscosity_model = build_viscosity_model(gas_model)
    thermal_conductivity_model = build_thermal_conductivity_model(gas_model)
    return TransportPropertyModel(viscosity_model, thermal_conductivity_model)


class IOFormat(Enum):
    NativeText = "native_text"
    NativeBinary = "native_binary"
    VtkText = "vtk_text"
    VtkBinary = "vtk_binary"


def string_to_io_format(string):
    if string == IOFormat.NativeText.value:
        return IOFormat.NativeText
    elif string == IOFormat.NativeBinary.value:
        return IOFormat.NativeBinary
    elif string == IOFormat.VtkText.value:
        return IOFormat.VtkText
    elif string == IOFormat.VtkBinary.value:
        return IOFormat.VtkBinary
    else:
        raise ValidationException(f"Unkown IO format {string}")


class IO:
    _json_values = ["flow_format",]
    __slots__ = _json_values
    _defaults_file = "io.json"

    def __init__(self, **kwargs):
        json_data = read_defaults(DEFAULTS_DIRECTORY,
                                  self._defaults_file)

        for key in self._json_values:
            setattr(self, key, string_to_io_format(json_data[key]))

        for key in kwargs:
            if type(kwargs[key]) is IOFormat:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, string_to_io_format(kwargs[key]))

    def validate(self):
        if (self.flow_format is IOFormat.VtkText or
                self.flow_format is IOFormat.VtkBinary):
            validation_errors.append(
                ValidationException("Vtk not supported as input format")
            )

    def as_dict(self):
        return {"flow_format": self.flow_format.value}


class Config:
    _json_values = ["convective_flux", "viscous_flux", "solver", "grid",
                    "gas_model", "transport_properties", "io"]
    __slots__ = _json_values

    def __init__(self):
        self.convective_flux = ConvectiveFlux()
        self.viscous_flux = ViscousFlux()
        self.solver = make_default_solver()
        self.gas_model = default_gas_model()
        self.transport_properties = build_transport_property_model(
            self.gas_model
        )
        self.io = IO()

    def validate(self):
        for setting in self.__slots__:
            getattr(self, setting).validate()
        if validation_errors:
            raise ValidationException(validation_errors)

    def write(self, directories):
        # directories to place things in
        config_directory = directories["config_dir"]
        config_file = directories["config_file"]
        config_file = f"{config_directory}/{config_file}"

        binary = self.io.flow_format is IOFormat.NativeBinary

        # extract all the values to go in the json config file
        json_values = {}
        for setting in self._json_values:
            json_values[setting] = getattr(self, setting).as_dict()

        # write the json config file
        with open(config_file, "w") as f:
            json.dump(json_values, f, indent=4)

        # write the grid files
        grid_directory = directories["grid_dir"]
        flow_directory = directories["flow_dir"]
        self.grid.write(grid_directory, flow_directory, binary)


def main(file_name, res_dir):
    # make the config directory
    global DEFAULTS_DIRECTORY
    global SHARE_DIRECTORY
    DEFAULTS_DIRECTORY = f"{res_dir}/defaults"
    SHARE_DIRECTORY = res_dir
    directories = read_defaults(DEFAULTS_DIRECTORY,
                                "directories.json")
    config_dir = directories["config_dir"]
    config_status = directories["config_status"]
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)

    # write a file saying the config is incomplete
    with open(f"{config_dir}/{config_status}", "w") as status:
        status.write("False")

    # build a namespace to execute the user supplied preparation script with
    config = Config()
    namespace = {
        "config": config,
        "ConvectiveFlux": ConvectiveFlux,
        "ViscousFlux": ViscousFlux,
        "Ausmdv": Ausmdv,
        "Hanel": Hanel,
        "Ldfss2": Ldfss2,
        "Block": Block,
        "Solver": Solver,
        "FlowState": FlowState,
        "GasState": GasState,
        "GasModel": GasModel,
        "IdealGas": IdealGas,
        "RungeKutta": RungeKutta,
        "SteadyState": SteadyState,
        "Gmres": Gmres,
        "IO": IO,
        "IOFormat": IOFormat,
        "supersonic_inflow": supersonic_inflow,
        "boundary_layer_inflow": boundary_layer_inflow,
        "supersonic_outflow": supersonic_outflow,
        "slip_wall": slip_wall,
        "adiabatic_no_slip_wall": adiabatic_no_slip_wall,
        "fixed_temperature_no_slip_wall": fixed_temperature_no_slip_wall,
        "subsonic_inflow": subsonic_inflow,
        "subsonic_outflow": subsonic_outflow,
        "BarthJespersen": BarthJespersen,
        "Unlimited": Unlimited,
        "ThermoInterp": ThermoInterp,
    }

    # run the user supplied script
    with open(file_name, "r") as f:
        exec(f.read(), namespace)

    # validate the configuration supplied by the user, and hopefully write
    # the configuration to file.
    config.validate()
    config.write(directories)
    with open(f"{config_dir}/{config_status}", "w") as status:
        status.write("True")
