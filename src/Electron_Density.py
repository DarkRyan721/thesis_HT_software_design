import numpy as np
from dolfinx import fem, io, mesh
from petsc4py import PETSc
import ufl
from ufl import SpatialCoordinate, exp, sqrt
import pyvista as pv
from dolfinx.geometry import BoundingBoxTree, compute_colliding_cells
from scipy.stats import gamma


def generate_density(domain, r0=0.04, sigma_r=0.007, A=1.2e15, z_min=-0.004, k=2, theta=0.012):
    """
    Retorna una Function FEniCSx con la densidad electrónica n_e(r,z)
    tipo anillo elongado, usando perfil radial gaussiano y axial gamma.
    """

    # Crear espacio de funciones
    V = fem.functionspace(domain, ("CG", 1))
    x = SpatialCoordinate(domain)

    # Coordenadas r y z (r en plano xy, z es eje axial)
    r = sqrt(x[0]**2 + x[1]**2)
    z = x[2]

    # --- PERFIL RADIAL (gaussiano centrado en r0)
    perfil_r = exp(-((r - r0)**2) / (2 * sigma_r**2))

    # --- PERFIL AXIAL (gamma convertido a expresión numérica)
    # Usamos numpy + scipy fuera de UFL para precomputar gamma
    # porque FEniCS no tiene gamma.pdf directamente
    def gamma_custom(z_val):
        return gamma.pdf(z_val, a=k, scale=theta, loc=z_min)

    # Interpolamos evaluando sobre los puntos de interpolación de V
    x_points = V.element.interpolation_points()
    ne_vals = np.zeros(len(x_points))

    for i, pt in enumerate(x_points):
        r_val = np.sqrt(pt[0]**2 + pt[1]**2)
        z_val = pt[2]
        val_r = np.exp(-((r_val - r0)**2) / (2 * sigma_r**2))
        val_z = gamma_custom(z_val)
        ne_vals[i] = A * val_r * val_z

    # Crear Function y asignar valores interpolados
    n_e = fem.Function(V)
    n_e.interpolate(lambda x: np.array([
        A * np.exp(-((np.sqrt(xi[0]**2 + xi[1]**2) - r0)**2) / (2 * sigma_r**2)) *
        gamma_custom(xi[2])
        for xi in x.T]).reshape((1, -1)))

    return n_e


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
