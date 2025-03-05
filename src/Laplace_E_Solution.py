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

# Obtiene las coordenadas de cada nodo de la malla en XYZ
X = domain.geometry.x  

# Extrae los valores de campo electrico en Ex, Ey y Ez para organizarlos en un vector de 3xn
E_values = E_field.x.array.reshape(-1, domain.geometry.dim).T

# Matriz con coordenadas y campo eléctrico [X,Y,Z,Ex,Ey,Ez]
E_np = np.hstack((X, E_values.T))

print(E_values[:,4000:4050])

#print(E_np.shape)
#_________________________________________________________________________________________________________
#                                       MATPLOTLIB NODOS CAMPO ELECTRICO

# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.cm as cm
# import matplotlib.colors as colors

# # Extraer coordenadas y componentes del campo eléctrico
# X, Y, Z = E_np[:, 0], E_np[:, 1], E_np[:, 2]  # Coordenadas nodales
# Ex, Ey, Ez = E_np[:, 3], E_np[:, 4], E_np[:, 5]  # Componentes del campo

# # Calcular la magnitud del campo eléctrico
# E_magnitude = np.sqrt(Ex**2 + Ey**2 + Ez**2)

# # Reducir la cantidad de vectores para mejorar visualización
# step = max(1, len(X) // 1000)  # Selecciona un subconjunto de puntos
# X, Y, Z = X[::step], Y[::step], Z[::step]
# Ex, Ey, Ez = Ex[::step], Ey[::step], Ez[::step]
# E_magnitude = E_magnitude[::step]

# # Normalizar la magnitud para asignar colores
# norm = colors.Normalize(vmin=np.min(E_magnitude), vmax=np.max(E_magnitude))
# cmap = cm.plasma  # Puedes cambiar a 'jet', 'viridis', etc.
# color_map = cmap(norm(E_magnitude))  # Convertir magnitudes a colores RGBA

# # Crear figura en 3D
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Graficar vectores con color por magnitud
# quiver = ax.quiver(X, Y, Z, Ex, Ey, Ez, length=0.1, normalize=True, cmap=cmap, array=E_magnitude)

# # Agregar nodos más gruesos con los mismos colores
# ax.scatter(X, Y, Z, c=E_magnitude, cmap=cmap, s=20, edgecolor="black")  # s controla el tamaño

# # Agregar barra de color para referencia
# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
# cbar.set_label("Magnitud del Campo Eléctrico |E|")

# # Configurar etiquetas y título
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("Campo Electrostático con Color por Magnitud")

# # Mostrar la gráfica
# plt.show()

#_________________________________________________________________________________________________________
#                                 MATPLOTLIB VECTORES CAMPO ELECTRICO

# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.cm as cm
# import matplotlib.colors as colors

# # Extraer coordenadas y componentes del campo eléctrico
# X, Y, Z = E_np[:, 0], E_np[:, 1], E_np[:, 2]  # Coordenadas nodales
# Ex, Ey, Ez = E_np[:, 3], E_np[:, 4], E_np[:, 5]  # Componentes del campo

# # Calcular la magnitud del campo eléctrico en cada nodo
# E_magnitude = np.sqrt(Ex**2 + Ey**2 + Ez**2)

# # Evitar división por cero para normalización
# E_magnitude[E_magnitude == 0] = 1

# # Calcular la dirección normalizada del campo eléctrico
# E_unit_x = Ex / E_magnitude
# E_unit_y = Ey / E_magnitude
# E_unit_z = Ez / E_magnitude

# # Si E_unit_z es 0, le asignamos 0.5 para que sea visible
# E_unit_z[E_unit_z == 0] = -0.2


# # Reducir la cantidad de vectores para mejor visualización
# step = max(1, len(X) // 500)  # Muestra un máximo de 500 vectores
# X, Y, Z = X[::step], Y[::step], Z[::step]
# E_unit_x, E_unit_y, E_unit_z = E_unit_x[::step], E_unit_y[::step], E_unit_z[::step]
# E_magnitude = E_magnitude[::step]

# # Normalizar la magnitud para usar en el colormap
# norm = colors.Normalize(vmin=np.min(E_magnitude), vmax=np.max(E_magnitude))
# cmap = cm.viridis  # Cambia a 'jet', 'viridis', etc.
# color_map = cmap(norm(E_magnitude))  # Convertir magnitudes a colores RGBA

# # Crear figura en 3D
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Graficar vectores normalizados con color según la magnitud
# quiver = ax.quiver(X, Y, Z, E_unit_x, E_unit_y, E_unit_z, length=10, normalize=True, colors=color_map)

# # Agregar nodos con colores y tamaño más grande
# ax.scatter(X, Y, Z, c=E_magnitude, cmap=cmap, s=30, edgecolor="black")

# # Agregar barra de color para referencia
# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
# cbar.set_label("Magnitud del Campo Eléctrico |E|")

# # Configurar etiquetas y título
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("Dirección del Campo Electrostático con Color por Magnitud")

# # Mostrar la gráfica
# plt.show()

#_________________________________________________________________________________________________________