"""
Ejemplo: Resolver ecuación de drift-diffusion para densidad electrónica n_e:
  -div( D*grad(n_e) - mu*n_e*E ) = f
en un dominio con malla generada en Gmsh.
"""

import os
import numpy as np
from mpi4py import MPI

# FEniCSx:
from dolfinx import fem, io, mesh
import ufl
from dolfinx.fem.petsc import LinearProblem

# Para resolver lineal
from petsc4py import PETSc

# --------------------------------------------------------------------
# Carga el campo E = -grad(phi) si ya lo resolviste con tu script de Laplace,
# o si lo tienes en un .npy con [x,y,z, Ex,Ey,Ez], etc.
# Para ilustrar, asumimos que E_field_np = np.load("Electric_Field_np.npy")
# --------------------------------------------------------------------
def load_electric_field(filename="Electric_Field_np.npy"):
    """
    Carga un array Nx6: [x, y, z, Ex, Ey, Ez]
    y devuelve una función de interpolación E(x).
    Aquí usaremos un approach simplificado de nearest neighbor
    o algo similar. En la práctica, convendría
    interpolar en la malla FEniCS.
    """
    data = np.load(filename)
    # data.shape = (N, 6)
    coords = data[:, :3]
    Evals  = data[:, 3:]
    # Podrías crear un cKDTree y hacer interpolation
    # Para ejemplo, devolvemos un callback "E(x) ~ 0"
    def E_func(x_):
        # x_.shape = (3, num_points)
        # Retornar un array (3, num_points) con E en cada punto
        # Ejemplo: E=0
        return np.zeros_like(x_)
    return E_func

def main():
    # 1) Cambiamos directorio para leer la malla en data_files
    os.chdir("data_files")

    # 2) Leemos la malla "SimulationZone.xdmf"
    with io.XDMFFile(MPI.COMM_WORLD, "SimulationZone.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="SPT100_Simulation_Zone")
        cell_tags = xdmf.read_meshtags(domain, name="SPT100_Simulation_Zone_cells")
        domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
        facet_tags = xdmf.read_meshtags(domain, name="SPT100_Simulation_Zone_facets")


    domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)

    # Info de la malla
    if domain.comm.rank == 0:
        print("Malla cargada. Dim:", domain.topology.dim)
        print("Nodos:", len(domain.geometry.x))

    # 3) Cargar o definir el campo E
    E_callable = load_electric_field("Electric_Field_np.npy")
    # Nota: en la práctica, interpolarías E en la malla FEniCS.

    # 4) Definir coeficientes (D_e, mu_e):
    D_e = 1e-3   # difusividad
    mu_e = 1e-3  # movilidad
    #  f: fuente (ionización - recombinación). Para ejemplo, f=0
    f_expr = fem.Constant(domain, PETSc.ScalarType(0.0))

    # 5) Espacio de funciones CG(1) para n_e
    #V = fem.FunctionSpace(domain, ("CG", 1))
    V = fem.functionspace(domain, ("CG", 1))

    #V = fem.FunctionSpace(domain, ("CG", 1, (domain.geometry.dim,)))


    # 6) n_e (desconocida) y v (función test)
    n_e = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    # 7) Definir E como ufl.Expression (depende de x).
    # Supongamos que E_field es un VEC CG(1):
    # V_vec = fem.VectorFunctionSpace(domain, ("CG", 1))
    V_vec = fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))

    E_fenics = fem.Function(V_vec)

    # Interpola a partir de datos => approach naive
    # 1) Crear un k-d tree con coords de .npy
    from scipy.spatial import cKDTree
    data = np.load("Electric_Field_np.npy")
    coords_data = data[:, :3]
    vals_data   = data[:, 3:]
    tree = cKDTree(coords_data)

    # 2) Para cada dof en E_fenics, buscar la "vecina" en coords_data
    x_dofs = V_vec.tabulate_dof_coordinates()  # shape (num_dofs, 3)
    for i, x_ in enumerate(x_dofs):
        _, idx = tree.query(x_)
        E_fenics.x.array[3*i  :3*i+3] = vals_data[idx]
    E_vec_ufl = ufl.as_vector((E_fenics[0], E_fenics[1], E_fenics[2]))

    # PDE: -div(D_e grad(n_e) - mu_e * n_e * E_vec) = f
    # Expandido => D_e * inner(grad(n_e), grad(v)) - mu_e * <(n_e E_vec)·grad(v)>
    # Ojo con signs y "drift-diffusion" forms.
    # a(n_e,v) = ∫ [D_e*(grad(n_e)·grad(v)) - mu_e*(n_e*(E_vec·grad(v))] dx
    # L(v)     = ∫ f_expr * v * dx
    a = (
        D_e * ufl.inner(ufl.grad(n_e), ufl.grad(v))
        - mu_e * ufl.inner(n_e * E_fenics, ufl.grad(v))
    ) * ufl.dx

    L = f_expr * v * ufl.dx

    # 8) Condiciones de frontera para n_e
    #    EJEMPLO: suponer n_e=0 en la pared o en un tag X, y libre en otras.
    #    Mirar 'facet_tags' y decide en qué tag pones Dirichlet
    boundary_conditions = []

    # supón que Walls tienen tag=5 => n_e=0:
    bc_value = fem.Function(V)
    bc_value.x.array[:] = 0.0
    walls_tag = 5
    walls_facets = facet_tags.indices[facet_tags.values==walls_tag]
    dofs_walls = fem.locate_dofs_topological(V, domain.topology.dim-1, walls_facets)
    bc_walls = fem.dirichletbc(bc_value, dofs_walls)
    boundary_conditions.append(bc_walls)

    # Ejemplo: Supón que en "Gas_inlet" (tag=3) n_e=1e17
    bc_value_inlet = fem.Function(V)
    bc_value_inlet.x.array[:] = 1e17

    inlet_tag = 3
    inlet_facets = facet_tags.indices[facet_tags.values==inlet_tag]
    dofs_inlet = fem.locate_dofs_topological(V, domain.topology.dim-1, inlet_facets)
    bc_inlet = fem.dirichletbc(bc_value_inlet, dofs_inlet)
    boundary_conditions.append(bc_inlet)

    # 9) Armar y resolver el problema lineal
    problem = LinearProblem(a, L, bcs=boundary_conditions,
                           petsc_options={"ksp_type":"preonly","pc_type":"lu"})
    n_e_sol = problem.solve()

    # 10) Guardar la solución
    n_e_sol.name = "electron_density"
    with io.XDMFFile(domain.comm, "Electron_Density.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(n_e_sol)

    if domain.comm.rank==0:
        print("Solución de densidad electrónica guardada en Electron_Density.xdmf.")

if __name__=="__main__":
    main()
