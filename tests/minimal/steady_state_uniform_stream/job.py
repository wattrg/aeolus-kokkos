gas_model = IdealGas(species = "air")
gas_state = GasState()
gas_state.T = 300
gas_state.rho = 1.0
gas_model.update_thermo_from_rhoT(gas_state)
flow_state = FlowState(gas=gas_state, vx=10.0, vy=0.0)

inflow_gs = GasState()
inflow_gs.T = 310
inflow_gs.rho = 1.0
gas_model.update_thermo_from_rhoT(inflow_gs)
initial = FlowState(gas=inflow_gs, vx=10.0, vy=0.0)

config.convective_flux = ConvectiveFlux(
    flux_calculator=Hanel(),
    reconstruction_order=1
)

config.gas_model = gas_model

config.solver = SteadyState(
    cfl=0.5,
    max_steps=5,
    plot_frequency=1,
    print_frequency=10,
    diagnostics_frequency=1,
    linear_solver=Gmres(max_iters=9*4, tol=1e-14)
)


# config.solver = RungeKutta(
#     cfl = 0.5,
#     max_step=100,
#     max_time=1.0,
#     plot_every_n_steps=1,
#     print_frequency=1
# )

config.grid = Block(
    file_name="grid.su2",
    initial_condition=initial,
    boundaries={
        "left": supersonic_inflow(flow_state),
        "top": supersonic_outflow(),
        # "bottom": supersonic_inflow(flow_state),
        "bottom": supersonic_outflow(),
        "right": supersonic_outflow()
    }
)
