gas_model = IdealGas(R=2.87103245856353623822e+02)
gas_state = GasState()
gas_state.T = 300
gas_state.rho = 1.0
gas_model.update_thermo_from_rhoT(gas_state)
flow_state = FlowState(gas=gas_state, vx=1000.0, vy=0.0)

inflow_gs = GasState()
inflow_gs.T = 310
inflow_gs.rho = 1.0
gas_model.update_thermo_from_rhoT(inflow_gs)
initial = FlowState(gas=inflow_gs, vx=1000.0, vy=0.0)

config.convective_flux = ConvectiveFlux(
    flux_calculator=Hanel(),
    reconstruction_order=1
)

config.gas_model = gas_model

config.solver = SteadyState(
    cfl=1.0,
    max_steps=1000,
    plot_frequency=10,
    print_frequency=100,
    diagnostics_frequency=1,
    linear_solver=Gmres(max_iters=36, tol=1e-14)
)


# config.solver = RungeKutta(
#     method="euler",
#     cfl = 1.0,
#     max_step=1,
#     max_time=1.0,
#     plot_every_n_steps=1,
#     print_frequency=1
# )

config.grid = Block(
    file_name="grid.su2",
    initial_condition=initial,
    boundaries={
        "left": supersonic_inflow(flow_state),
        "right": supersonic_outflow(),
        "top": slip_wall(),
        "bottom": slip_wall(),
        # "top": supersonic_outflow(),
        # "bottom": supersonic_outflow(),
    }
)
