from dolfinx import mesh, io
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner
from dolfinx import fem
from petsc4py import PETSc
from dolfinx.fem import FunctionSpace

with io.XDMFFile(MPI.COMM_WORLD, "SimulationZone.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="SPT100_Simulation_Zone")  

domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

print("Malla cargada exitosamente en FEniCSx")
print(f"Dimensión de la malla: {domain.topology.dim}")
print(f"Número de nodos: {domain.geometry.x.shape[0]}")


with io.XDMFFile(MPI.COMM_WORLD, "SimulationZone.xdmf", "r") as xdmf:
    facet_tags = xdmf.read_meshtags(domain, name="SPT100_Simulation_Zone_facets")
    cell_tags = xdmf.read_meshtags(domain,name = "SPT100_Simulation_Zone_cells")

print("Etiquetas de frontera cargadas exitosamente.")
print(f"Número de facetas etiquetadas: {len(facet_tags.values)}")
print(f"Etiquetas disponibles: {set(facet_tags.values)}")


V = fem.functionspace(domain, ("CG", 1))

u_bc = fem.Function(V)
u_bc.x.array[:] = np.nan

for tag, value in [(3, 1.0), (4, 0.0)]:
    dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facet_tags.indices[facet_tags.values == tag])
    u_bc.x.array[dofs] = value

boundary_conditions = []
for tag in [3, 4]:
    dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facet_tags.indices[facet_tags.values == tag])
    bc = fem.dirichletbc(u_bc, dofs)
    boundary_conditions.append(bc)


with io.XDMFFile(MPI.COMM_WORLD, "boundary_conditions.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain) 
    xdmf.write_function(u_bc)  

print("Archivo 'boundary_conditions.xdmf' generado correctamente.")


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

with io.XDMFFile(domain.comm, "out_poisson/poisson.xdmf", "w") as file:
    file.write_mesh(domain)
    file.write_function(uh)

try:
    import pyvista

    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh_poisson.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")