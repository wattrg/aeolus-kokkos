gas_model = IdealGas(species="air")
gas_state = GasState()
gas_state.p = 1.013e3
gas_state.T = 300.0
gas_model.update_thermo_from_pT(gas_state)
vx = 4 * gas_model.speed_of_sound(gas_state)
inflow = FlowState(gas=gas_state, vx=vx)
initial = FlowState(gas=gas_state, vx=vx)

config.convective_flux = ConvectiveFlux(
    flux_calculator=Ausmdv(),
    reconstruction_order=1,
)

config.viscous_flux = ViscousFlux(enabled = True)

config.gas_model = gas_model

config.solver = SteadyState(
    # cfl=LinearInterpolateCfl(times=[0, 10, 100], cfls=[10.0, 10.0, 10000.0]),
    cfl=ResidualBasedCfl(growth_threshold=1000, power=0.95, start_cfl=1.0),
    max_steps=10000,
    print_frequency=50,
    plot_frequency=500,
    diagnostics_frequency=1,
    tolerance=1e-10,
    linear_solver=FGmres(
        tolerance=1e-2,
        max_iters=50,
        preconditioner_tolerance=1e-2,
        max_preconditioner_iters=10
    )
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
