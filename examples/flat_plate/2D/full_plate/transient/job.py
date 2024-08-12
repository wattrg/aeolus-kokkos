gas_model = IdealGas(species="air")
gas_state = GasState()
gas_state.p = 1.013e3
gas_state.T = 300.0
gas_model.update_thermo_from_pT(gas_state)
vx = 4 * gas_model.speed_of_sound(gas_state)
inflow = FlowState(gas=gas_state, vx=vx)
initial = FlowState(gas=gas_state, vx=vx)
max_time = 3 * 1.0 / vx

config.convective_flux = ConvectiveFlux(
    flux_calculator=Ausmdv(),
    reconstruction_order=2,
)

config.viscous_flux = ViscousFlux(enabled = True)

config.gas_model = gas_model

config.solver = RungeKutta(
    method="ssp-rk3",
    cfl = 2.0,
    max_step = 500000,
    max_time = max_time,
    plot_every_n_steps = -20,
    plot_frequency = max_time / 10,
    print_frequency = 1000,
)

config.grid = Block(
    file_name="grid.su2",
    initial_condition=inflow,
    boundaries = {
        "inflow": supersonic_inflow(inflow),
        "outflow": supersonic_outflow(),
        "wall": fixed_temperature_no_slip_wall(temperature = 300),
    }
)
