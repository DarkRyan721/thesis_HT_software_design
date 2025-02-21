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
import basix

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

Volt = 10000

boundary_conditions = []
for tag, value in [(3, Volt), (4, 0.0)]:  # Potencial fijo en facetas 3 y 4
    dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1,facet_tags.indices[facet_tags.values == tag])
    u_bc.x.array[dofs] = value
    bc = fem.dirichletbc(u_bc, dofs)
    boundary_conditions.append(bc)

with io.XDMFFile(MPI.COMM_WORLD, "boundary_conditions.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain) 
    xdmf.write_function(u_bc)  

phi = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = inner(grad(phi), grad(v)) * dx
L = fem.Constant(domain, PETSc.ScalarType(0)) * v * dx #inner(f, v) * dx + inner(g, v) * ds
problem = LinearProblem(a, L, bcs=boundary_conditions, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
phi_h = problem.solve()


with io.XDMFFile(domain.comm, "Laplace.xdmf", "w") as file:
    file.write_mesh(domain)
    file.write_function(phi_h)

#u_sol = fem.Function(V)
#u_sol.interpolate(phi_h)
#E_field = -grad(u_sol)


V_vector = fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))

E_field = fem.Function(V_vector)

E_field_expr = fem.Expression(-grad(phi_h), V_vector.element.interpolation_points())

E_field.interpolate(E_field_expr)

print(type(E_field))

with io.XDMFFile(domain.comm, "Electric_Field.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(E_field)
