gas_model = IdealGas(species = "air")
gas_state = GasState()
gas_state.T = 300
gas_state.rho = 1.225
gas_model.update_thermo_from_rhoT(gas_state)
flow_state = FlowState(gas=gas_state, vx=1000.0, vy=500.0)

config.convective_flux = ConvectiveFlux(
    flux_calculator=Hanel(),
    reconstruction_order=1
)

config.gas_model = gas_model

config.solver = SteadyState(
    cfl=0.5,
    max_steps=10,
    plot_frequency=1,
    print_frequency=1,
    diagnostics_frequency=1
)

config.grid = Block(
    file_name="grid.su2",
    initial_condition=flow_state,
    boundaries={
        "left": supersonic_inflow(flow_state),
        "top": supersonic_outflow(),
        "bottom": supersonic_inflow(flow_state),
        "right": supersonic_outflow()
    }
)
