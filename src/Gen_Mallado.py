#-------LIBRERIAS-------
import gmsh
import numpy as np
import sys
from mpi4py import MPI
from dolfinx.io import XDMFFile
import dolfinx.io.gmshio as gmshio

#----------GENERADO DE MALLA------------

def create_mesh(comm: MPI.Comm, model: gmsh.model, name: str, filename: str, mode: str):
    """
    Convierte un modelo de Gmsh a una malla de dolfinx y la guarda en formato XDMF.
    
    Parámetros:
    -----------
    comm : MPI.Comm
        Comunicador MPI que se utiliza para distribuir la malla entre procesos.
    model : gmsh.model
        Modelo geométrico de Gmsh que se convertirá en malla.
    name : str
        Nombre lógico de la malla, para referencia interna.
    filename : str
        Nombre del archivo XDMF donde se guardará la malla.
    mode : str
        Modo de escritura (por ejemplo "w" para escribir la primera vez 
        o "a" para agregar información en un archivo existente).
    """
    # Convirtiendo el modelo de Gmsh a una malla de dolfinx con etiquetas de células (ct) y facetas (ft).
    msh, ct, ft = gmshio.model_to_mesh(model, comm, rank=0)
    
    # Asignar nombres para la malla y para las etiquetas de células y facetas.
    msh.name = name
    ct.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"

    # Crear conectividad 2D-3D (facetas->volúmenes) necesaria para escribir correctamente las facetas.
    msh.topology.create_connectivity(2, 3)

    # Guardar la malla y las etiquetas (meshtags) en el archivo XDMF.
    with XDMFFile(msh.comm, filename, mode) as file:
        file.write_mesh(msh)
        file.write_meshtags(
            ct, msh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )
        file.write_meshtags(
            ft, msh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )


def solicitar_dimensiones():
    """
    Solicita al usuario los valores de radio externo, radio interno y profundidad
    del canal (en mm), verificando que los datos sean enteros.

    Retorna:
    --------
    (r_ext, r_int, profundidad) : tupla de int
        - r_ext: Radio externo.
        - r_int: Radio interno.
        - profundidad: Longitud del canal (altura).
    """
    while True:
        try:
            r_ext = int(input("Ingresa el radio externo (en mm): "))
            r_int = int(input("Ingresa el radio interno (en mm): "))
            profundidad = int(input("Ingresa la profundidad del canal (en mm): "))
            
            # Validación 1: radio interno < radio externo.
            if r_int >= r_ext:
                print("El radio interno debe ser menor que el radio externo. Intenta de nuevo.\n")
                continue
            
            # Validación 2: todos los valores deben ser positivos.
            if r_ext <= 0 or r_int <= 0 or profundidad <= 0:
                print("Todos los valores deben ser positivos. Intenta de nuevo.\n")
                continue
            
            break  # Si se cumplen las validaciones, salimos del bucle.

        except ValueError:
            print("Error: Por favor ingresa un número entero válido.\n")
    
    return r_ext, r_int, profundidad


def main():
    """
    Función principal que:
      1. Inicializa Gmsh.
      2. Solicita al usuario los parámetros geométricos (radios y altura).
      3. Construye la geometría (un cubo grande + cilindro hueco).
      4. Genera y etiqueta la malla en Gmsh.
      5. Exporta la malla en formato .msh y .xdmf.
      6. Finaliza la sesión de Gmsh.
    """
    # Inicializa Gmsh y crea un nuevo modelo:
    gmsh.initialize()
    gmsh.model.add("SPT100_Simulation_Zone")
    
    # Solicita al usuario los valores de radio externo, interno y profundidad:
    #R_big, R_small, H = solicitar_dimensiones()
    R_big = 0.1/2
    R_small = 0.056/2
    H = 0.02

    # Se define un parámetro adicional para generar un cubo que representará la "pluma" (dominio externo).
    # Aquí usamos, por ejemplo, un cubo de lado 3 * R_big. 
    L = 2 * R_big  # Lado del cubo de la pluma

    # (Opcional) Guardar parámetros en un archivo .txt para uso posterior:
    with open("data_files/geometry_parameters.txt", "w") as f:
        f.write(f"radio_externo: {R_big}\n")
        f.write(f"radio_interno: {R_small}\n")
        f.write(f"profundidad: {H}\n")
        f.write(f"lado_cubo: {L}\n")

    # -------------------------------------------------------------------------
    # Construcción de la geometría:
    #
    #  - Un cubo de dimensiones L x L x L, que empieza en z=H, para simular la 
    #    región exterior (plume).
    #  - Dos cilindros concéntricos (uno externo y uno interno) de altura H, 
    #    para representar el propulsor y su canal hueco.
    # -------------------------------------------------------------------------

    # 1) Crear el cubo:
    cube = gmsh.model.occ.addBox(-L/2, -L/2, H, L, L, L)

    # 2) Crear dos cilindros (externo e interno) que parten de z=0 a z=H:
    cylinder_outer = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, H, R_big)
    cylinder_inner = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, H, R_small)

    # 3) "Cortar" el cilindro interno del externo para crear una pared hueca:
    cylinder = gmsh.model.occ.cut([(3, cylinder_outer)], [(3, cylinder_inner)])
    gmsh.model.occ.synchronize()

    # 4) Fragmentar el cubo con el cilindro hueco para generar volúmenes separados:
    fragmented = gmsh.model.occ.fragment([(3, cube)], cylinder[0])
    gmsh.model.occ.synchronize()

    # Identificamos los volúmenes resultantes de la fragmentación.
    # Normalmente, se asume:
    #   - Volumen 1: el cubo fragmentado (Plume_Domain).
    #   - Volumen 2: el cilindro (Thruster_Domain).
    cube_volume = 1
    cylinder_volume = 2

    # 5) Crear grupos físicos para cada volumen (esto permite asignarles 
    #    etiquetas identificables en la malla):
    gmsh.model.addPhysicalGroup(3, [cube_volume], 1)
    gmsh.model.setPhysicalName(3, 1, "Plume_Domain")

    gmsh.model.addPhysicalGroup(3, [cylinder_volume], 2)
    gmsh.model.setPhysicalName(3, 2, "Thruster_Domain")

    # -------------------------------------------------------------------------
    # Etiquetado de superficies (fronteras) para condiciones de contorno:
    # -------------------------------------------------------------------------
    surfaces = gmsh.model.get_entities(dim=2)  # Entidades de dimensión 2 (caras)

    inlet_surfaces = []
    outlet_thruster_surfaces = []
    cylinder_wall_surfaces = []
    outlet_plume_surfaces = []
    ids_oultet =[]

    # Definimos una tolerancia para comparar coordenadas con isclose.
    tol = 1e-3  

    for surface in surfaces:
        # Obtener centro de masa de la superficie
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])

        # Si la cara está cerca de z=0 -> entrada de gas
        if np.isclose(com[2], 0, atol=tol):
            inlet_surfaces.append(surface[1])

        # Si la cara está cerca de z=H -> salida del propulsor
        elif np.isclose(com[2], H, atol=tol):
            outlet_thruster_surfaces.append(surface[1])
        # Si la cara está cerca de x=±L/2, y=±L/2, o z=H+L -> salida de la pluma
        elif (np.isclose(com[0],  L/2, atol=tol) or np.isclose(com[0], -L/2, atol=tol) or
              np.isclose(com[1],  L/2, atol=tol) or np.isclose(com[1], -L/2, atol=tol) or
              np.isclose(com[2], H + L, atol=tol)):
            outlet_plume_surfaces.append(surface[1])

        # Lo que no encaja en las categorías anteriores se asume que es pared
        else:
            cylinder_wall_surfaces.append(surface[1])

    # Ajustes forzados (hardcode) según la geometría específica que generó Gmsh:
    # Se fuerza a agregar la superficie con ID = 18 a la salida de la pluma.
    # Esto depende de la numeración interna de Gmsh, que puede variar.
    # Ojo: Usar con precaución, pues en otras versiones de Gmsh podrían cambiar los IDs.
    outlet_plume_surfaces.append(5)
    outlet_plume_surfaces.append(6)
    outlet_plume_surfaces.append(8)
    outlet_plume_surfaces.append(9)
    outlet_plume_surfaces.append(10)

    # Filtramos la salida del propulsor para quedarnos solo con la superficie ID = 12.
    outlet_thruster_surfaces = [s for s in outlet_thruster_surfaces if s == 7]

    # 6) Crear grupos físicos para las fronteras:
    gmsh.model.addPhysicalGroup(2, inlet_surfaces, 3)
    gmsh.model.setPhysicalName(2, 3, "Gas_inlet")

    gmsh.model.addPhysicalGroup(2, outlet_thruster_surfaces, 4)
    gmsh.model.setPhysicalName(2, 4, "Thruster_outlet")

    gmsh.model.addPhysicalGroup(2, cylinder_wall_surfaces, 5)
    gmsh.model.setPhysicalName(2, 5, "Walls")

    gmsh.model.addPhysicalGroup(2, outlet_plume_surfaces, 6)
    gmsh.model.setPhysicalName(2, 6, "Plume_outlet")

    # -------------------------------------------------------------------------
    # Parámetros de discretización de la malla
    # -------------------------------------------------------------------------
    # El tamaño mínimo y máximo de los elementos.
    debye_lenght = 0.00001
    lambda_factor = 100000
    element_size = debye_lenght*lambda_factor
    
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.003/2) #0.01
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.003)

    # Generar la malla 3D.
    gmsh.model.mesh.generate(3)

    # Exportar la malla en formato Gmsh .msh
    gmsh.write("data_files/SimulationZone.msh")

    # Exportar la malla a formato XDMF para FEniCSx:
    create_mesh(MPI.COMM_WORLD, gmsh.model, 
                name="SPT100_Simulation_Zone",
                filename="data_files/SimulationZone.xdmf",
                mode="w")

    # Finaliza la sesión de Gmsh (muy importante para liberar recursos).
    gmsh.finalize()


if __name__ == "__main__":
    main()

