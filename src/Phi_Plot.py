
import numpy as np
from dolfinx import fem, io, mesh
from petsc4py import PETSc
import ufl
from ufl import SpatialCoordinate, exp, sqrt
import pyvista as pv
from dolfinx.geometry import BoundingBoxTree, compute_colliding_cells



# Cargar puntos y densidad desde archivos numpy
E_np = np.load('data_files/Electric_Field_np.npy')
points = E_np[:, :3]  # X, Y, Z

n0 = np.load('data_files/phi_np.npy')  # Tu archivo de densidad

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