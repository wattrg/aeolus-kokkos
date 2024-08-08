mach = 3.0
T = 300
n_flows = 3
n_plots = 10
length = 1.0
gas_model = IdealGas(R = 287.0)
gas_state = GasState()
gas_state.rho = 1.225
gas_state.T = T
gas_model.update_thermo_from_rhoT(gas_state)
v = mach * gas_model.speed_of_sound(gas_state)
flow_state = FlowState(gas=gas_state, vz=v)

config.convective_flux = ConvectiveFlux(
    flux_calculator = Ausmdv(),
    reconstruction_order = 2
)

config.gas_model = gas_model

config.solver = RungeKutta(
    method = "ssp-rk3",
    cfl = 3.0,
    max_step = 100000,
    max_time = n_flows * length / v,
    plot_every_n_steps = -1,
    plot_frequency = n_flows / n_plots * length / v,
    print_frequency = 500
)

config.grid = Block(
    file_name="grid.su2", 
    initial_condition=flow_state,
    boundaries = {
        "inflow": supersonic_inflow(flow_state),
        "outflow": supersonic_outflow(),
        "ramp": slip_wall(),
        "symmetry": slip_wall(),
        "sides": slip_wall(),
        "top": supersonic_outflow(),
    }
)
