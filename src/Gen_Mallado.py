import gmsh
import sys
import meshio
import numpy as np
import pyvista as pv

# Función para leer puntos desde un archivo STEP
def extract_step_points(step_file):
    gmsh.initialize()
    gmsh.model.add("ImportedModel")

    # Importar el archivo STEP
    gmsh.model.occ.importShapes(step_file)
    gmsh.model.occ.synchronize()

    # Obtener todas las entidades de puntos (dim=0)
    points = gmsh.model.getEntities(0)  

    point_coords = []
    for point in points:
        x, y, z = gmsh.model.getValue(point[0], point[1], [])  # Obtener coordenadas
        point_coords.append((x, y, z))

    gmsh.finalize()
    return point_coords

# Función para graficar los puntos con etiquetas
def plot_step_points_with_labels(points):
    """
    Función para graficar puntos extraídos de un archivo STEP usando PyVista, con etiquetas para cada punto.
    
    Parámetros:
    - points: Lista de tuplas (x, y, z) con las coordenadas de los puntos.
    """
    if not points:
        print("No hay puntos para graficar.")
        return

    # Convertir la lista de puntos en un array de NumPy
    points_array = np.array(points)

    # Crear la nube de puntos en PyVista
    point_cloud = pv.PolyData(points_array)

    # Configurar la visualización
    plotter = pv.Plotter()
    plotter.add_points(point_cloud, color="red", point_size=5)

    # Agregar etiquetas con nombres a cada punto
    for i, (x, y, z) in enumerate(points):
        plotter.add_point_labels(np.array([[x, y, z]]), [f"Punto {i+1}"], font_size=10, point_color="blue", text_color="black", point_size=10)

    plotter.show()

# Nombre del archivo STEP (Cambia esto por el archivo correcto)
step_filename = "SPT-100.step"

# Obtener coordenadas de los puntos
points = extract_step_points(step_filename)

# Mostrar coordenadas de los puntos
print("Coordenadas de los puntos en el archivo STEP:")
for i, (x, y, z) in enumerate(points):
    print(f"Punto {i+1}: ({x:.3f}, {y:.3f}, {z:.3f})")

# Graficar los puntos con etiquetas
plot_step_points_with_labels(points)


gmsh.initialize()
gmsh.model.add("SPT100_Simulation_Zone")

# Crear el cubo
L = 1.0  # Longitud del cubo
cube = gmsh.model.occ.addBox(0, 0, 0, L, L, L)

# Crear el cilindro hueco
R_big = 0.3
R_small = 0.1
H = 1.0
pos_x, pos_y, pos_z = L / 2, L / 2, L

cylinder_outer = gmsh.model.occ.addCylinder(pos_x, pos_y, pos_z, 0, 0, H, R_big)
cylinder_inner = gmsh.model.occ.addCylinder(pos_x, pos_y, pos_z, 0, 0, H, R_small)
cylinder = gmsh.model.occ.cut([(3, cylinder_outer)], [(3, cylinder_inner)])

gmsh.model.occ.synchronize()

# Fragmentar para hacer la malla compatible
fragmented = gmsh.model.occ.fragment([(3, cube)], cylinder[0])
gmsh.model.occ.synchronize()

print(cylinder)
cube_volume = 1
cylinder_volume= 2

gmsh.model.addPhysicalGroup(3, [cube_volume], 1) 
gmsh.model.setPhysicalName(3, 1, "Plume_Domain")
gmsh.model.addPhysicalGroup(3, [cylinder_volume], 2)  
gmsh.model.setPhysicalName(3, 2 ,"Thruster_Domain")

surfaces = gmsh.model.get_entities(2)

inlet_surfaces = []
outlet_thruster_surfaces = []
cylinder_wall_surfaces = []
outlet_plume_surfaces = []

tol = 1e-3

for surface in surfaces:
    com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])

    if np.isclose(com[2],(L+H),atol=tol):
        inlet_surfaces.append(surface[1])

    elif np.isclose(com[2],L,atol=tol):
        outlet_thruster_surfaces.append(surface[1])
    
    elif np.isclose(com[0], L, atol=tol) or np.isclose(com[0], 0, atol=tol) or np.isclose(com[2], 0, atol=tol) or np.isclose(com[1],0,atol=tol) or np.isclose(com[1],L,atol=tol):
        outlet_plume_surfaces.append(surface[1])

    else:
        cylinder_wall_surfaces.append(surface[1])

    #print("Informacion del centro de masa", com)
outlet_plume_surfaces.append(19)
outlet_thruster_surfaces = [s for s in outlet_thruster_surfaces if s == 13]

print("Arreglo Outlet:",outlet_thruster_surfaces)
gmsh.model.addPhysicalGroup(2, inlet_surfaces, 3)  # Entrada del propulsor
gmsh.model.setPhysicalName(2,3,"Gas_inlet")
gmsh.model.addPhysicalGroup(2, outlet_thruster_surfaces, 4)  # Salida del dominio
gmsh.model.setPhysicalName(2,4,"Thruster_outlet")
gmsh.model.addPhysicalGroup(2, cylinder_wall_surfaces, 5)  # Paredes internas del cilindro
gmsh.model.setPhysicalName(2,5,"Walls")
gmsh.model.addPhysicalGroup(2,outlet_plume_surfaces,6) #Salida del plume
gmsh.model.setPhysicalName(2,6,"Plume_outlet")

# Aumentar la resolución de la malla
gmsh.option.setNumber("Mesh.MeshSizeMin", R_big / 10)
gmsh.option.setNumber("Mesh.MeshSizeMax", R_big / 5)

# Generar la malla
gmsh.model.mesh.generate(3)
gmsh.write("SimulationMesh.msh")

# Mostrar en Gmsh
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()



# Leer el archivo .msh
msh = meshio.read("SimulationMesh.msh")

# Revisar los tipos de celdas presentes
print("Tipos de celdas en el archivo:", msh.cells_dict.keys())

# Verificar si hay elementos tetraédricos
if "tetra" not in msh.cells_dict:
    raise ValueError("No se encontraron elementos tetraédricos en el archivo .msh")

# Crear la malla en formato adecuado para FEniCSx
mesh = meshio.Mesh(
    points=msh.points.astype("float64"),  # FEniCSx espera float64
    cells=[("tetra", msh.cells_dict["tetra"])]
)

# Guardar en formato XDMF (con HDF5)
meshio.write("SimulationMesh.xdmf", mesh)


with open("SimulationMesh.xdmf", "r") as file:
    print("Contenido del XDMF:\n", file.read())
