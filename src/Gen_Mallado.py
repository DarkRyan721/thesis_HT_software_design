import gmsh
import numpy as np
import meshio
import sys
import pyvista as pv

def extract_cylinder_radii(step_file):
    gmsh.initialize()
    gmsh.model.add("ImportedModel")

    # Importar el archivo STEP
    gmsh.model.occ.importShapes(step_file)
    gmsh.model.occ.synchronize()

    # Obtener todas las superficies (dim=2)
    surfaces = gmsh.model.getEntities(2)
    xmaxes = []
    ymaxes = []

    for surface in surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(surface[0], surface[1])
        xmaxes.append(xmax)
        ymaxes.append(ymax)

    gmsh.finalize
    return max_values(xmaxes,ymaxes)

def max_values(xmaxes,ymaxes):
    # Eliminar duplicados y ordenar en orden descendente
    unique_xmaxes = sorted(set(xmaxes), reverse=True)
    unique_ymaxes = sorted(set(ymaxes), reverse=True)

    # Definir valores por defecto en caso de que no se encuentren suficientes datos
    rad_int_cyl = None
    rad_ext_smallcyl = None
    positive_ymaxes = np.abs(unique_ymaxes)

    lenght = max(positive_ymaxes)

    # Obtener el segundo valor más grande si existen al menos dos valores únicos
    if len(unique_xmaxes) >= 2:
        rad_int_cyl = unique_xmaxes[1]

    # Obtener el menor valor mayor a 10
    for value in sorted(unique_xmaxes):
        if value > 10:
            rad_ext_smallcyl = value
            break
    
    
        

    return rad_int_cyl, rad_ext_smallcyl, lenght

step_filename = "SPT-100(2).step"
# Obtener los valores procesados
rad_int_cyl, rad_ext_smallcyl, lenght = extract_cylinder_radii(step_filename)



gmsh.initialize()
gmsh.model.add("SPT100_Simulation_Zone")

R_big = rad_int_cyl#0.3
R_small = rad_ext_smallcyl #0.1
# Crear el cubo
L = 3*R_big  # Longitud del cubo
cube = gmsh.model.occ.addBox(0, 0, 0, L, L, L)

# Crear el cilindro hueco
H = lenght
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

