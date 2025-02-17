import meshio

mesh_from_gmsh = meshio.read("SimulationMesh.msh")
cells = mesh_from_gmsh.cells_dict
physical_tags = mesh_from_gmsh.cell_data_dict["gmsh:physical"]

mesh = meshio.Mesh(
    points=mesh_from_gmsh.points,
    cells=[("tetra", cells["tetra"])],  
    cell_data={"name_to_read": [physical_tags["tetra"]]},  
)

meshio.write("SimulationMesh.xdmf", mesh)

facet_mesh = meshio.Mesh(
    points=mesh_from_gmsh.points,
    cells=[("triangle", cells["triangle"])], 
    cell_data={"name_to_read": [physical_tags["triangle"]]},
)

meshio.write("SimulationMesh_facet.xdmf", facet_mesh)

print("Malla exportada con éxito a SimulationMesh.xdmf y SimulationMesh_facet.xdmf")
