import numpy as np
import pyvista as pv

# Cargar datos
E_np = np.load('data_files/Electric_Field_np.npy')
points = E_np[:, :3]  # X, Y, Z
vectors = E_np[:, 3:]  # Ex, Ey, Ez

# Calcular la magnitud para el mapa de calor
magnitudes = np.linalg.norm(vectors, axis=1)
log_magnitudes = np.log10(magnitudes + 1e-3)  # Evitamos log(0)

# Crear un objeto PolyData
mesh = pv.PolyData(points)

# Añadir los vectores al objeto
mesh["vectors"] = vectors
mesh["magnitude"] = log_magnitudes

# Crear un glyph (flecha por vector)
glyphs = mesh.glyph(orient="vectors", scale=False, factor=0.0025)

# Crear el plotter
plotter = pv.Plotter()
plotter.set_background("white")
plotter.add_mesh(glyphs, scalars="magnitude", cmap="plasma")
#plotter.add_scalar_bar(title="|E| [V/m]", vertical=True)
plotter.add_axes()
plotter.add_title("Campo Eléctrico - Dirección y Magnitud")

plotter.show()
