import gmsh
import numpy as np
import sys
from mpi4py import MPI
from dolfinx.io import XDMFFile
import dolfinx.io.gmshio as gmshio



def create_mesh(comm: MPI.Comm, model: gmsh.model, name: str, filename: str, mode: str):

    msh, ct, ft = gmshio.model_to_mesh(model, comm, rank=0)
    msh.name = name
    ct.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"
    with XDMFFile(msh.comm, filename, mode) as file:
        msh.topology.create_connectivity(2, 3)
        file.write_mesh(msh)
        file.write_meshtags(
            ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )
        file.write_meshtags(
            ft, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )

def extract_cylinder_radii(step_file):
    gmsh.initialize()
    gmsh.model.add("ImportedModel")

    gmsh.model.occ.importShapes(step_file)
    gmsh.model.occ.synchronize()

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
    unique_xmaxes = sorted(set(xmaxes), reverse=True)
    unique_ymaxes = sorted(set(ymaxes), reverse=True)

    rad_int_cyl = None
    rad_ext_smallcyl = None
    positive_ymaxes = np.abs(unique_ymaxes)

    lenght = max(positive_ymaxes)

    if len(unique_xmaxes) >= 2:
        rad_int_cyl = unique_xmaxes[1]

    for value in sorted(unique_xmaxes):
        if value > 10:
            rad_ext_smallcyl = value
            break
    
    
        

    return rad_int_cyl, rad_ext_smallcyl, lenght

step_filename = "SPT-100(2).step"
rad_int_cyl, rad_ext_smallcyl, lenght = extract_cylinder_radii(step_filename)

gmsh.initialize()
gmsh.model.add("SPT100_Simulation_Zone")

R_big = rad_int_cyl
R_small = rad_ext_smallcyl
L = 3*R_big
cube = gmsh.model.occ.addBox(0, 0, 0, L, L, L)

H = lenght
pos_x, pos_y, pos_z = L / 2, L / 2, L

cylinder_outer = gmsh.model.occ.addCylinder(pos_x, pos_y, pos_z, 0, 0, H, R_big)
cylinder_inner = gmsh.model.occ.addCylinder(pos_x, pos_y, pos_z, 0, 0, H, R_small)
cylinder = gmsh.model.occ.cut([(3, cylinder_outer)], [(3, cylinder_inner)])

gmsh.model.occ.synchronize()

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



outlet_plume_surfaces.append(19)
outlet_thruster_surfaces = [s for s in outlet_thruster_surfaces if s == 13]

print("Arreglo Outlet:",outlet_thruster_surfaces)
gmsh.model.addPhysicalGroup(2, inlet_surfaces, 3)  
gmsh.model.setPhysicalName(2,3,"Gas_inlet")
gmsh.model.addPhysicalGroup(2, outlet_thruster_surfaces, 4) 
gmsh.model.setPhysicalName(2,4,"Thruster_outlet")
gmsh.model.addPhysicalGroup(2, cylinder_wall_surfaces, 5)
gmsh.model.setPhysicalName(2,5,"Walls")
gmsh.model.addPhysicalGroup(2,outlet_plume_surfaces,6)
gmsh.model.setPhysicalName(2,6,"Plume_outlet")

gmsh.option.setNumber("Mesh.MeshSizeMin", R_big / 10)
gmsh.option.setNumber("Mesh.MeshSizeMax", R_big / 5)

gmsh.model.mesh.generate(3)

#gmsh.write("SimulationZone.msh")

create_mesh(MPI.COMM_WORLD, gmsh.model, "SPT100_Simulation_Zone", "SimulationZone.xdmf","w")

gmsh.finalize()


