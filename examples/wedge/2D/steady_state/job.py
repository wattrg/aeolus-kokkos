mach = 3.0
T = 300
n_flows = 3
n_plots = 10
length = 1.0
gas_model = IdealGas(R=287.0)
gas_state = GasState()
gas_state.rho = 1.225
gas_state.T = T
gas_model.update_thermo_from_rhoT(gas_state)
vx = mach * gas_model.speed_of_sound(gas_state)
flow_state = FlowState(gas=gas_state, vx=vx)

config.convective_flux = ConvectiveFlux(
    flux_calculator=Hanel(),
    reconstruction_order=1
)

config.gas_model = gas_model

config.solver = SteadyState(
    cfl=0.5,
    max_steps=10,
    print_frequency=1,
    plot_frequency=1,
    diagnostics_frequency=1,
    tolerance=1e-2,
    linear_solver=Gmres(tol=1e-2, max_iters=2)
)

config.grid = Block(
    file_name="grid.su2",
    initial_condition=flow_state,
    boundaries={
        "inflow": supersonic_inflow(flow_state),
        "outflow": supersonic_outflow(),
        "wall": slip_wall(),
    }
)

config.io = IO(flow_format="native_text")
