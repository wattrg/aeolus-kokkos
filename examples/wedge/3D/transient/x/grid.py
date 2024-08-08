import gmsh
import math

gmsh.initialize()

# general settings
gmsh.option.setNumber("General.Terminal", 1)
# gmsh.option.setNumber("Mesh.Algorithm", 8)  # quads
# gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.SaveAll", 1)

# points
inflow_length = 0.2
ramp_length = 0.8
inflow_height = 0.2
outflow_height = 0.5
theta = 10
depth = 0.1
ramp_height = math.tan(math.radians(theta))

grid_size = depth / 3

# points
a00 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, grid_size)
b00 = gmsh.model.geo.add_point(inflow_length, 0.0, 0.0, grid_size)
c00 = gmsh.model.geo.add_point(inflow_length+ramp_length, ramp_height, 0.0,
                               grid_size)
a10 = gmsh.model.geo.add_point(0.0, inflow_height, 0.0, grid_size)
b10 = gmsh.model.geo.add_point(inflow_length, inflow_height, 0.0, grid_size)
c10 = gmsh.model.geo.add_point(inflow_length+ramp_length,
                               ramp_height+outflow_height,
                               0.0, grid_size)

# lines
a00b00 = gmsh.model.geo.add_line(a00, b00)
b00c00 = gmsh.model.geo.add_line(b00, c00)
a00a10 = gmsh.model.geo.add_line(a00, a10)
a10a00 = gmsh.model.geo.add_line(a10, a00)
b00b10 = gmsh.model.geo.add_line(b00, b10)
c00c10 = gmsh.model.geo.add_line(c00, c10)
c10c00 = gmsh.model.geo.add_line(c10, c00)
b10a10 = gmsh.model.geo.add_line(b10, a10)
c10b10 = gmsh.model.geo.add_line(c10, b10)

# curve loops
loop = gmsh.model.geo.add_curve_loop([a00b00, b00c00, c00c10, c10b10, b10a10, a10a00])

# surfaces
surface = gmsh.model.geo.add_plane_surface([loop])

# volumes
volume = gmsh.model.geo.extrude([(2, surface)], 0, 0, depth)

# boundaries
gmsh.model.geo.add_physical_group(2, [40], name="inflow")
gmsh.model.geo.add_physical_group(2, [28], name="outflow")
gmsh.model.geo.add_physical_group(2, [24], name="ramp")
gmsh.model.geo.add_physical_group(2, [20], name="symmetry")
gmsh.model.geo.add_physical_group(2, [32, 36], name="top")
gmsh.model.geo.add_physical_group(2, [1, 41], name="sides")

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(3)

# gmsh.fltk.run()

gmsh.write("grid.su2")
