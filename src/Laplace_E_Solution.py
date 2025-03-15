"""
--------------------------------------------------------------------------------
Script: Solución de la ecuación de Laplace (potencial eléctrico) en el dominio
        generado para el SPT-100, y cálculo y visualización del campo eléctrico.
--------------------------------------------------------------------------------
Este script:
  1. Carga la malla en formato XDMF (SimulationZone.xdmf) generada con Gmsh.
  2. Carga las etiquetas de facetas y volúmenes (facet_tags, cell_tags).
  3. Define un espacio de funciones (V) para el potencial eléctrico.
  4. Aplica condiciones de frontera Dirichlet (en facetas con tags 3 y 4).
  5. Resuelve la ecuación de Laplace: -∇·∇φ = 0.
  6. Calcula el campo eléctrico como E = -∇φ.
  7. Visualiza los vectores del campo eléctrico en 3D usando matplotlib.

Requisitos:
  - FEniCSx (dolfinx)
  - mpi4py
  - numpy
  - petsc4py
  - matplotlib (para la visualización)
  - Gmsh (sólo si se modifican o regeneran las mallas)
  - Archivo "SimulationZone.xdmf" que contenga la malla y etiquetas.

Modo de uso:
  1. Ajustar y asegurarse de que "SimulationZone.xdmf" y "boundary_conditions.xdmf"
     existan y sean accesibles.
  2. Ejecutar con: python3 <nombre_de_este_script>.py

--------------------------------------------------------------------------------
"""

# Importaciones principales
from mpi4py import MPI
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Se requiere para la proyección 3D
import matplotlib.cm as cm
import matplotlib.colors as colors

# Importaciones de dolfinx y UFL
from dolfinx import fem, io, mesh
import ufl
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

# Importaciones de petsc4py
from petsc4py import PETSc


os.chdir("data_files")

# -----------------------------------------------------------------------------
# Lectura de la malla y de las etiquetas desde el archivo XDMF
# -----------------------------------------------------------------------------
with io.XDMFFile(MPI.COMM_WORLD, "SimulationZone.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="SPT100_Simulation_Zone")  

# Se crea la conectividad entre facetas (dim-1) y celdas (dim)
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

# Información básica sobre la malla
print("Malla cargada exitosamente en FEniCSx")
print(f"Dimensión de la malla: {domain.topology.dim}")
print(f"Número de nodos: {domain.geometry.x.shape[0]}")

# Se leen las etiquetas de facetas y celdas (para condiciones de frontera y dominios)
with io.XDMFFile(MPI.COMM_WORLD, "SimulationZone.xdmf", "r") as xdmf:
    facet_tags = xdmf.read_meshtags(domain, name="SPT100_Simulation_Zone_facets")
    cell_tags = xdmf.read_meshtags(domain, name="SPT100_Simulation_Zone_cells")

print("Etiquetas de frontera cargadas exitosamente.")
print(f"Número de facetas etiquetadas: {len(facet_tags.values)}")
print(f"Etiquetas disponibles: {set(facet_tags.values)}")


# -----------------------------------------------------------------------------
# Definición del espacio de funciones y condiciones de frontera
# -----------------------------------------------------------------------------
# Se define un espacio de funciones de orden 1 (polinomios CG)
V = fem.functionspace(domain, ("CG", 1))

# Se crea una función para almacenar los valores de potencial en la frontera
u_bc = fem.Function(V)
u_bc.x.array[:] = np.nan  # Inicialmente NaN para diferenciar regiones

# Valor de potencial fijado en la frontera
Volt = 10000

# Definición de condiciones de Dirichlet en las facetas etiquetadas
#  - Tag 3 tendrá potencial de 10kV
#  - Tag 4 tendrá potencial de 0V
boundary_conditions = []
for tag, value in [(3, Volt), (4, 0.0)]:
    # Ubica los dofs en las facetas con etiqueta `tag`
    dofs = fem.locate_dofs_topological(
        V, domain.topology.dim - 1, facet_tags.indices[facet_tags.values == tag]
    )
    # Asigna el valor a esos dofs
    u_bc.x.array[dofs] = value
    bc = fem.dirichletbc(u_bc, dofs)
    boundary_conditions.append(bc)

# Se almacena la función de frontera en un archivo XDMF (para visualización/depuración)
with io.XDMFFile(MPI.COMM_WORLD, "boundary_conditions.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain) 
    xdmf.write_function(u_bc)


# -----------------------------------------------------------------------------
# Formulación de la ecuación de Laplace y resolución
# -----------------------------------------------------------------------------
# Sea phi la variable a resolver y v el test function
phi = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# a(phi, v) = ∫(∇phi · ∇v) dx
a = inner(grad(phi), grad(v)) * dx

# L = 0 en todo el dominio -> ∫(0 * v) dx = 0
L = fem.Constant(domain, PETSc.ScalarType(0)) * v * dx

# Se crea y resuelve el problema lineal
problem = LinearProblem(
    a,
    L,
    bcs=boundary_conditions,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
phi_h = problem.solve()

# Se escribe la solución (potencial) en un archivo XDMF
with io.XDMFFile(domain.comm, "Laplace.xdmf", "w") as file:
    file.write_mesh(domain)
    file.write_function(phi_h)


# -----------------------------------------------------------------------------
# Cálculo del campo eléctrico E = -∇φ
# -----------------------------------------------------------------------------
# Se crea un espacio de funciones vectorial, 
# con la misma familia pero de dimensión igual a dim del dominio.
V_vector = fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))
E_field = fem.Function(V_vector)

# Definición de la expresión para el campo: E = -grad(phi_h)
E_field_expr = fem.Expression(-grad(phi_h), V_vector.element.interpolation_points())
E_field.interpolate(E_field_expr)

print("Tipo de objeto E_field:", type(E_field))

# Se escribe el campo eléctrico en un archivo XDMF
with io.XDMFFile(domain.comm, "Electric_Field.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(E_field)


# -----------------------------------------------------------------------------
# Extracción de datos nodales y visualización del campo eléctrico
# -----------------------------------------------------------------------------
# Obtiene las coordenadas (XYZ) de los nodos de la malla
X = domain.geometry.x

# El campo E_field.x.array contiene los valores del campo eléctrico en cada dof
# Se reordena y se separa en Ex, Ey, Ez
E_values = E_field.x.array.reshape(-1, domain.geometry.dim).T

# Se concatena la información de coordenadas y campo eléctrico en un array (N x 6)
# [x, y, z, Ex, Ey, Ez]
E_np = np.hstack((X, E_values.T))

np.save("Electric_Field_np.npy", E_np)

# -----------------------------------------------------------------------------
# Visualización 3D con matplotlib
# -----------------------------------------------------------------------------
# Extraemos coordenadas y componentes del campo
X, Y, Z = E_np[:, 0], E_np[:, 1], E_np[:, 2]   # Coordenadas
Ex, Ey, Ez = E_np[:, 3], E_np[:, 4], E_np[:, 5]  # Componentes de E

# Calculamos la magnitud del campo eléctrico
E_magnitude = np.sqrt(Ex**2 + Ey**2 + Ez**2)

# Evitamos división por cero para normalización
E_magnitude[E_magnitude == 0] = 1

# Calculamos la dirección normalizada del campo
E_unit_x = Ex / E_magnitude
E_unit_y = Ey / E_magnitude
E_unit_z = Ez / E_magnitude

# Ajuste opcional para E_unit_z = 0 (por ejemplo, para hacerlo visible)
E_unit_z[E_unit_z == 0] = -0.2

# Reducimos la cantidad de vectores para evitar sobrecarga visual
step = max(1, len(X) // 500)  # Máximo de 500 vectores
X, Y, Z = X[::step], Y[::step], Z[::step]
E_unit_x = E_unit_x[::step]
E_unit_y = E_unit_y[::step]
E_unit_z = E_unit_z[::step]
E_magnitude = E_magnitude[::step]

# Normalizamos valores para aplicarlos a un colormap
norm = colors.Normalize(vmin=np.min(E_magnitude), vmax=np.max(E_magnitude))
cmap = cm.viridis
color_map = cmap(norm(E_magnitude))

# Creamos la figura 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Graficamos los vectores normalizados con color según la magnitud
quiver = ax.quiver(
    X, Y, Z,
    E_unit_x, E_unit_y, E_unit_z,
    length=10,  # Controla la longitud real de las flechas
    normalize=True,
    colors=color_map
)

# Graficamos los nodos como puntos
ax.scatter(X, Y, Z, c=E_magnitude, cmap=cmap, s=30, edgecolor="black")

# Agregamos una barra de color que indique la magnitud
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
cbar.set_label("Magnitud del Campo Eléctrico |E|")

# Etiquetas de los ejes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Dirección del Campo Eléctrico con Color por Magnitud")

# Se muestra la gráfica
plt.show()
