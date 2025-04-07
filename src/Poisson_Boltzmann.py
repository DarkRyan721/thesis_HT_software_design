import os
import numpy as np
from mpi4py import MPI
from dolfinx import fem, io, mesh
import ufl
from ufl import ds, dx, grad, inner
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc


def solve_potential(domain, facet_tags, n0_array, n_i=1e10, n_i_final=1e17, n_steps=10, volt_tag=3, ground_tag=4, Volt=300):
    # --- Parámetros iniciales ---
    e = 1.602e-19  # Carga elemental (C)
    epsilon_0 = 8.854e-12  # Permitividad eléctrica en el vacío (F/m)
    k_B = 1.38e-23  # Constante de Boltzmann (J/K)
    T = 300  # Temperatura en Kelvin (ajustar según tu caso)

    # Crear espacio de funciones
    V = fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))

    # Inicializar la densidad de electrones
    n0 = n0_array

    # Crear la densidad de iones uniforme
    ni = np.zeros(len(n0))

    # Asignar valor uniforme en la región Z [0, 0.02]
    coordinates = domain.geometry.x
    for i, coord in enumerate(coordinates):
        if 0 <= coord[2] <= 0.02:
            ni[i] = n_i  # Puedes ajustar esto si quieres usar un incremento gradual

    # --- Parámetros de convergencia ---
    tolerance = 1e-6
    max_iter = 100
    converged = False

    # Potencial inicial
    phi = fem.Function(V)
    phi.interpolate(lambda x: np.full(x.shape[1], 10.0))  

    # Inicialización de condiciones de contorno
    boundary_conditions = []

    for tag, value in [(volt_tag, Volt), (ground_tag, 0.0)]:
        dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facet_tags.indices[facet_tags.values == tag])
        u_bc = fem.Function(V)
        u_bc.x.array[dofs] = value
        bc = fem.dirichletbc(u_bc, dofs)
        boundary_conditions.append(bc)

    # --- Iteraciones de Poisson-Boltzmann ---
    for iteration in range(max_iter):
        # Actualizar densidad de electrones con Boltzmann
        phi_array = phi.x.array  # Potencial actual
        ne = n0 * np.exp(-e * phi_array / (k_B * T))

        # Calcular la densidad de carga total
        rho = e * (ni - ne)

        # Crear función de densidad de carga
        rho_func = fem.Function(V)
        np.copyto(rho_func.x.array, rho)

        # Definir la forma débil
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = -rho_func * v * ufl.dx / epsilon_0

        # Resolver problema
        problem = fem.petsc.LinearProblem(a, L, bcs=boundary_conditions)
        phi_new = problem.solve()

        # Verificar convergencia
        diff = np.linalg.norm(phi_new.x.array - phi.x.array)
        print(f"Iteración {iteration}: Error = {diff}")

        if diff < tolerance:
            converged = True
            break

        # Actualizar phi para la siguiente iteración
        phi.x.array[:] = phi_new.x.array[:]

    if not converged:
        print("El algoritmo no convergió dentro del número máximo de iteraciones.")
    else:
        print("Convergencia alcanzada.")

    # Exportar phi a un archivo .npy
    phi_values = phi.x.array
    np.save("phi_np.npy", phi_values)
    print("Archivo 'phi_np.npy' guardado exitosamente.")

    return phi

def compute_electric_field(domain, phi_h):
    """
    Calcula el campo eléctrico E = -∇φ y lo devuelve como una dolfinx.fem.Function.

    Parámetros:
    -----------
    domain : dolfinx.mesh.Mesh
        Malla cargada en FEniCSx.
    phi_h : dolfinx.fem.Function
        Función solución para el potencial eléctrico.

    Retorna:
    --------
    E_field : dolfinx.fem.Function
        Campo eléctrico, un vector en cada nodo.
    """
    # Espacio vectorial (dim igual a la dimensión espacial)
    V_vector = fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))

    # Definición de la expresión para E = -grad(phi_h)
    E_field = fem.Function(V_vector)
    E_field_expr = fem.Expression(-grad(phi_h), V_vector.element.interpolation_points())
    E_field.interpolate(E_field_expr)

    # Almacenamos el campo en un archivo XDMF
    with io.XDMFFile(domain.comm, "Electric_Field.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(E_field)

    return E_field


def main():
    os.chdir("data_files")  # Cambiar al directorio donde tienes tus archivos

    with io.XDMFFile(MPI.COMM_WORLD, "SimulationZone.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="SPT100_Simulation_Zone")
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

    with io.XDMFFile(MPI.COMM_WORLD, "SimulationZone.xdmf", "r") as xdmf:
        facet_tags = xdmf.read_meshtags(domain, name="SPT100_Simulation_Zone_facets")
        cell_tags = xdmf.read_meshtags(domain, name="SPT100_Simulation_Zone_cells")

    # Cargar el archivo numpy que contiene las densidades iniciales n0
    n0_array = np.load("density_n0.npy")  # Asegúrate de que este archivo tenga el tamaño adecuado

    if n0_array.ndim == 1:
        n0_array = n0_array.reshape(-1)

    # Resolver el problema de Poisson-Boltzmann
    phi_h = solve_potential(
        domain=domain,
        facet_tags=facet_tags,
        n0_array=n0_array,
        volt_tag=3,
        ground_tag=4,
        Volt=300
    )

    E_field = compute_electric_field(domain, phi_h)

    X = domain.geometry.x
    # El campo E_field.x.array contiene los valores del campo eléctrico en cada dof
    # Se reordena y se separa en Ex, Ey, Ez
    E_values = E_field.x.array.reshape(-1, domain.geometry.dim).T

    # Se concatena la información de coordenadas y campo eléctrico en un array (N x 6)
    # [x, y, z, Ex, Ey, Ez]
    E_np = np.hstack((X, E_values.T))
    np.save("Electric_Field_np.npy", E_np)


if __name__ == "__main__":
    main()

