-- corner.lua
--
-- Mach 2.0 flow expanding around a convex corner,
-- details are taken from the AIAA verification project.
--
-- Kyle A. Damm [2022-08-01]
--

-- Try to get the file format from environment variable
fileFmt = os.getenv("LMR_FILE_FORMAT") or "rawbinary"

-- General settings
config.solver_mode = "steady"
config.dimensions = 2
config.axisymmetric = false
config.print_count = 1
config.save_residual_values = false
config.save_limiter_values = false
config.field_format = fileFmt
config.grid_format = fileFmt

grid0 = registerFluidGrid{
   grid=UnstructuredGrid:new{filename="grid.su2", fmt="su2text"},
   fsTag="initial",
   bcTags={top="outflow",bottom="outflow",left="inflow", right="outflow"}
}

-- ==========================================================
-- Freestream conditions
-- ==========================================================
nsp, nmodes, gm = setGasModel('ideal-air.gas')
gs = GasState:new{gm}
M_inf = 2.0
gs.rho  = 1.0 -- kg/m^3
gs.T  = 300.0 -- K
gm:updateThermoFromRHOT(gs)
inflow = FlowState:new{p=gs.p, T=gs.T, velx=10.0}

gs.T = 310.0
gm:updateThermoFromRHOT(gs)
initial = FlowState:new{p=gs.p, T=gs.T, velx=10.0}

-- ==========================================================
-- Block definitions
-- ==========================================================
flowDict = {}
flowDict["initial"] = initial
flowDict["inflow"] = inflow

bcDict = {
   inflow = InFlowBC_Supersonic:new{flowState=inflow},
   outflow = OutFlowBC_Simple:new{}
}

makeFluidBlocks(bcDict, flowDict)

-- ==========================================================
-- Solver configuration
-- ==========================================================

-- invsicid flux settings
config.flux_calculator= "hanel"
config.apply_entropy_fix = false
config.interpolation_order = 1
config.extrema_clipping = false
config.thermo_interpolator = "rhop"
config.apply_limiter = false

-- viscous flux settings
config.viscous = false

-- Set temporal integration settings
config.residual_smoothing = false

NewtonKrylovGlobalConfig{
   number_of_steps_for_setting_reference_residuals = 0,
   max_newton_steps = 5,
   stop_on_relative_residual = 1.0e-12,
   number_of_phases = 1,
   inviscid_cfl_only = true,
   use_line_search = false,
   use_physicality_check = false,
   max_linear_solver_iterations = 50,
   max_linear_solver_restarts = 0,
   use_scaling = true,
   frechet_derivative_perturbation = 1.0e-50,
   use_preconditioner = false,
   preconditioner_perturbation = 1.0e-50,
   preconditioner = "ilu",
   ilu_fill = 0,
   total_snapshots = 10,
   steps_between_snapshots = 1,
   steps_between_diagnostics = 1
}

NewtonKrylovPhase:new{
   residual_interpolation_order = 1,
   jacobian_interpolation_order = 1,
   frozen_limiter_for_jacobian = false,
   linear_solve_tolerance = 1e-14,
   use_auto_cfl = false,
   start_cfl = 0.5,
   max_cfl = 0.5,
}


