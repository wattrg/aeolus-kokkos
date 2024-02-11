import gmsh

gmsh.initialize()

# general settings
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.Algorithm", 8) # quads
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.SaveAll", 1)

# points
height = 0.00005
length = 0.0005
a = gmsh.model.geo.addPoint(0.0, 0.0, 0.0) # 1
b = gmsh.model.geo.addPoint(length, 0.0, 0.0) # 2
c = gmsh.model.geo.addPoint(length, height, 0.0) # 3
d = gmsh.model.geo.addPoint(0.0, height, 0.0) # 4

# lines
la = gmsh.model.geo.addLine(a, b)
lb = gmsh.model.geo.addLine(b, c)
lc = gmsh.model.geo.addLine(c, d)
ld = gmsh.model.geo.addLine(d, a)
cl = gmsh.model.geo.addCurveLoop([la, lb, lc, ld])

# surface
gmsh.model.geo.addPlaneSurface([cl])

# boundaries
gmsh.model.geo.addPhysicalGroup(1, [la], name="wall")
gmsh.model.geo.addPhysicalGroup(1, [ld], name="inflow")
gmsh.model.geo.addPhysicalGroup(1, [lb, lc], name="outflow")

# mesh size
size = 0.000001
gmsh.model.geo.mesh.setSize([(0, a), (0,b), (0,c), (0,d)], size)

# syncronise the geometry so we can mesh
gmsh.model.geo.synchronize()

boundary_layer = gmsh.model.mesh.field.add("BoundaryLayer")
gmsh.model.mesh.field.set_numbers(boundary_layer, "CurvesList", [la])
gmsh.model.mesh.field.set_numbers(boundary_layer, "PointsList", [a, b])
gmsh.model.mesh.field.set_number(boundary_layer, "SizeFar", size)
gmsh.model.mesh.field.set_number(boundary_layer, "Size", 0.1e-6)
gmsh.model.mesh.field.set_number(boundary_layer, "Quads", 1)
gmsh.model.mesh.field.set_number(boundary_layer, "Ratio", 1.03)
gmsh.model.mesh.field.set_number(boundary_layer, "Thickness", height/1.5)
gmsh.model.mesh.field.setAsBoundaryLayer(boundary_layer)
gmsh.model.geo.mesh.set_recombine(2, 1)


gmsh.model.geo.synchronize()
gmsh.model.mesh.optimize('', niter=100)

# generate the mesh
gmsh.model.mesh.generate(2)

# write the mesh
gmsh.write("grid.su2")

# open gmsh GUI to check the mesh
# gmsh.fltk.run()

gmsh.finalize()
