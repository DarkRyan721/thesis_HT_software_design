import numpy as np
from dolfinx import fem, io, mesh
from petsc4py import PETSc
import ufl
from ufl import SpatialCoordinate, exp, sqrt
import pyvista as pv
from dolfinx.geometry import BoundingBoxTree, compute_colliding_cells


def generate_density(domain, n0 = 1e16, R=0.035, z0=0.01, sigma=0.008):
    """
    Genera la densidad de electrones inicial n0 usando una distribución gaussiana tipo anillo.

    Parámetros:
    -----------
    domain : dolfinx.mesh.Mesh
        Malla cargada en FEniCSx.
    rho_value : float
        Carga total deseada [#particulas/m^3].
    R : float
        Radio medio del anillo [m].
    z0 : float
        Posición axial del anillo [m].
    sigma : float
        Ancho del anillo [m].
    epsilon_0 : float
        Permisividad del vacío.

    Retorna:
    --------
    n0 : dolfinx.fem.Function
        Densidad de carga generada interpolada en el dominio.
    """
    V = fem.functionspace(domain, ("CG", 1))
    x = SpatialCoordinate(domain)

    # Distribución gaussiana tipo anillo
    r = sqrt(x[0]**2 + x[1]**2)
    dist_sq = (r - R)**2 + (x[2] - z0)**2
    ring_expr = (n0) * exp(-dist_sq / (2 * sigma**2))

    # Interpolación de la densidad de carga
    n0 = fem.Function(V)
    n0.interpolate(fem.Expression(ring_expr, V.element.interpolation_points()))

    return n0


def save_density(n0, filename="density_n0.npy"):
    """
    Guarda la densidad de carga en un archivo .npy.
    """
    n0_array = n0.x.array
    np.save(filename, n0_array)


def plot_density(domain, title="Densidad de Carga Inicial n0"):
    # Cargar puntos y densidad desde archivos numpy
    E_np = np.load('Electric_Field_np.npy')
    points = E_np[:, :3]  # X, Y, Z

    n0 = np.load('density_n0.npy')  # Tu archivo de densidad

    # Crear un objeto PolyData
    mesh = pv.PolyData(points)
    mesh["n0"] = n0  # Añadir la densidad como point data

    # Crear el plotter
    plotter = pv.Plotter()
    plotter.set_background("black")  # Fondo negro

    # Añadir la malla con la densidad
    plotter.add_mesh(mesh, scalars="n0", cmap="plasma", point_size=5, 
                render_points_as_spheres=True,
                scalar_bar_args={
                    'title': "",
                    'color': 'white',  # Color de los números y título
                    'label_font_size': 12,
                    'title_font_size': 10
                })

    # Cambiar el color de los ejes a blanco
    plotter.add_axes(color="white")

    # Añadir la barra de color con texto blanco
    # Obtener la barra de color original generada por add_mesh()

    # Mostrar el título con letras blancas
    plotter.add_text("Mapa de Calor - Densidad de Carga Inicial n0", position='upper_edge', color="white", font_size=14)

    plotter.show()


if __name__ == "__main__":
    import os
    from mpi4py import MPI

    # Cambiar al directorio adecuado
    os.chdir("data_files")

    # Cargar la malla
    with io.XDMFFile(MPI.COMM_WORLD, "SimulationZone.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="SPT100_Simulation_Zone")

    # Generar la densidad de carga inicial
    n0 = generate_density(domain)

    # # Guardar la densidad de carga
    save_density(n0)

    plot_density(domain)

    print("Densidad de carga inicial n0 generada y guardada exitosamente como 'density_n0.npy'.")
