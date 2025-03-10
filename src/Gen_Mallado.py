"""
--------------------------------------------------------------------------------
Script: Generación de malla para SPT-100 (zona de simulación)
--------------------------------------------------------------------------------
Este script importa un archivo STEP de un SPT-100, extrae los radios interiores y
exteriores de los cilindros relevantes y construye una geometría en Gmsh para:
  - El dominio de la pluma (Plume_Domain).
  - El dominio del propulsor (Thruster_Domain).
  - Superficies con condiciones de frontera (Gas_inlet, Thruster_outlet, Walls, 
    Plume_outlet).

Finalmente, se genera la malla 3D y se exporta en dos formatos:
  - Formato .msh de Gmsh (SimulationZone.msh).
  - Formato .xdmf para FEniCSx (SimulationZone.xdmf).

Requisitos previos de Python:
  - gmsh
  - mpi4py
  - numpy
  - dolfinx
  - dolfinx.io.gmshio (que viene con dolfinx)

Requisitos del sistema:
  - Tener instalado Gmsh
  - Tener soporte para MPI (por ejemplo, mpich o openmpi)

Modo de uso (ejemplo):
  1. Ajustar el nombre del archivo STEP (step_filename).
  2. Ejecutar el script con un Python compatible (>=3.8).

--------------------------------------------------------------------------------
"""

import gmsh
import numpy as np
import sys
from mpi4py import MPI
from dolfinx.io import XDMFFile
import dolfinx.io.gmshio as gmshio


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
        Modo de escritura (por ejemplo "w" para escribir la primera vez o "a" para
        agregar información en un archivo existente).
    """
    # Converte el modelo de Gmsh a una malla para dolfinx junto con las etiquetas
    # de celdas (ct) y de facetas (ft).
    msh, ct, ft = gmshio.model_to_mesh(model, comm, rank=0)
    
    # Asigna nombres para la malla y para las etiquetas de celdas y facetas
    msh.name = name
    ct.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"

    # Escribe la malla y las etiquetas (meshtags) en un archivo .xdmf
    with XDMFFile(msh.comm, filename, mode) as file:
        # Crea la conectividad 2D-3D (facetas->volúmenes) antes de escribir
        msh.topology.create_connectivity(2, 3)
        file.write_mesh(msh)
        file.write_meshtags(
            ct, msh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )
        file.write_meshtags(
            ft, msh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )


def extract_cylinder_radii(step_file: str):
    """
    Importa un archivo STEP y obtiene los radios interior y exterior de los cilindros 
    así como la 'longitud' (basada en el bounding box máximo) de la geometría SPT-100.
    
    Parámetros:
    -----------
    step_file : str
        Nombre o ruta del archivo STEP a importar.
    
    Retorna:
    --------
    rad_int_cyl : float or None
        Radio interior del cilindro principal.
    rad_ext_smallcyl : float or None
        Radio exterior de otro cilindro relevante.
    lenght : float
        Longitud total basada en el tamaño máximo en el eje Y (aprox).
    """
    gmsh.initialize()
    gmsh.model.add("ImportedModel")

    # Importa la geometría STEP
    gmsh.model.occ.importShapes(step_file)
    gmsh.model.occ.synchronize()

    # Se asume que la geometría contiene superficies (2D) de las cuales se 
    # extraen bounding boxes
    surfaces = gmsh.model.getEntities(2)
    
    xmaxes = []
    ymaxes = []
    xminimos = []

    # Recorre cada superficie y extrae el bounding box
    for surface in surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(surface[0], surface[1])
        xmaxes.append(xmax)
        ymaxes.append(ymax)
        xminimos.append(xmin)

    # Cierra la sesión de gmsh (aunque no se llama "()" al final, 
    # se recomienda gmsh.finalize() con paréntesis para asegurarlo).
    gmsh.finalize

    # Procesa los valores máximos y mínimos para extraer radios y longitud
    return max_values(xmaxes, ymaxes, xminimos)


def max_values(xmaxes, ymaxes, xminimos):
    """
    Función auxiliar que, a partir de los valores en X y Y de bounding boxes,
    determina:
      - rad_int_cyl : Un posible radio interior (segundo más grande en Xmax).
      - rad_ext_smallcyl : Un posible radio exterior de cilindro más pequeño.
      - lenght : Tamaño máximo en Y (aprox la altura del cilindro).
    """
    # Ordena y quita duplicados
    unique_xmaxes = sorted(set(xmaxes), reverse=True)
    unique_ymaxes = sorted(set(ymaxes), reverse=True)
    unique_xminimos = sorted(set(xminimos), reverse=True)

    rad_int_cyl = None
    rad_ext_smallcyl = None

    # Se toman valores absolutos para evitar signos negativos
    positive_ymaxes = np.abs(unique_ymaxes)
    positive_xmins = np.abs(unique_xminimos)

    # Se asume que la longitud total está dada por la máxima magnitud en Y
    lenght = max(positive_ymaxes)

    # Si hay al menos 2 valores en la lista de xmax, asumimos el segundo
    # como radio interior.
    if len(unique_xmaxes) >= 2:
        rad_int_cyl = unique_xmaxes[1]
    
    # Para el radio exterior, buscamos el primer valor en 'unique_xmaxes'
    # que supere el mínimo valor absoluto de 'xmin'.
    for value in sorted(unique_xmaxes):
        if value > min(positive_xmins):
            rad_ext_smallcyl = value
            break

    return rad_int_cyl, rad_ext_smallcyl, lenght


# -----------------------------------------------------------------------------
# Main del script: construcción de la geometría y generación de malla
# -----------------------------------------------------------------------------
# Ajustar este nombre si el archivo STEP se encuentra en otra ruta o tiene 
# otro nombre.
step_filename = "SPT-100(2).step"

# Se extraen radios y la longitud de la geometría a partir del STEP
rad_int_cyl, rad_ext_smallcyl, lenght = extract_cylinder_radii(step_filename)

# Inicializa Gmsh para empezar a construir una nueva geometría
gmsh.initialize()
gmsh.model.add("SPT100_Simulation_Zone")

# Define parámetros geométricos clave
R_big = rad_int_cyl        # Radio principal del cilindro (externo)
R_small = rad_ext_smallcyl # Radio interno de otro cilindro
L = 3 * R_big              # Escala para el cubo
H = lenght                 # Altura (se toma el bounding box en Y como referencia)

# Crea un cubo con centro en (-L/2, -L/2) y altura H
#   - Origen en (-L/2, -L/2, H)
#   - Dimensiones L x L x L
cube = gmsh.model.occ.addBox(-L/2, -L/2, H, L, L, L)

# Crea dos cilindros (externo e interno) en la base (z=0) y con altura H
cylinder_outer = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, H, R_big)
cylinder_inner = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, H, R_small)

# Corta el cilindro interno del cilindro externo para obtener una 'pared'
cylinder = gmsh.model.occ.cut([(3, cylinder_outer)], [(3, cylinder_inner)])
gmsh.model.occ.synchronize()

# Fragmenta el cubo con el cilindro hueco, produciendo volúmenes separados
fragmented = gmsh.model.occ.fragment([(3, cube)], cylinder[0])
gmsh.model.occ.synchronize()

# Se sabe que:
#   - El volumen 1 corresponde al cubo fragmentado.
#   - El volumen 2 corresponde al cilindro (hueco).
cube_volume = 1
cylinder_volume = 2

# Crea grupos físicos (labels) para identificar cada volumen
gmsh.model.addPhysicalGroup(3, [cube_volume], 1)
gmsh.model.setPhysicalName(3, 1, "Plume_Domain")

gmsh.model.addPhysicalGroup(3, [cylinder_volume], 2)
gmsh.model.setPhysicalName(3, 2, "Thruster_Domain")

# Identifica las superficies (2D) para asignar condiciones de frontera
surfaces = gmsh.model.get_entities(2)

inlet_surfaces = []
outlet_thruster_surfaces = []
cylinder_wall_surfaces = []
outlet_plume_surfaces = []

tol = 1e-3  # Tolerancia para comparación de floats (center of mass)

for surface in surfaces:
    # Obtiene el centro de masa de la superficie
    com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])

    # Si la cara está cerca de z=0, se asigna como la entrada de gas
    if np.isclose(com[2], 0, atol=tol):
        inlet_surfaces.append(surface[1])

    # Si la cara está cerca de z=H, se asigna como la salida del propulsor
    elif np.isclose(com[2], H, atol=tol):
        outlet_thruster_surfaces.append(surface[1])
    
    # Si la cara está cerca de x=±L/2, y=±L/2, z=H+L, es la salida de la pluma
    elif (np.isclose(com[0], L/2, atol=tol) or np.isclose(com[0], -L/2, atol=tol) or
          np.isclose(com[2], L+H, atol=tol) or np.isclose(com[1], L/2, atol=tol) or
          np.isclose(com[1], -L/2, atol=tol)):
        outlet_plume_surfaces.append(surface[1])

    # Si no cumple ninguna anterior, se asume que es la pared del cilindro
    else:
        cylinder_wall_surfaces.append(surface[1])

# (Ajustes particulares) Se fuerza a agregar la superficie '18' como salida de pluma
outlet_plume_surfaces.append(18)

# Se filtra la salida del propulsor para quedarse solo con la superficie '12'
outlet_thruster_surfaces = [s for s in outlet_thruster_surfaces if s == 12]

# Agrupa las superficies en grupos físicos
gmsh.model.addPhysicalGroup(2, inlet_surfaces, 3)
gmsh.model.setPhysicalName(2, 3, "Gas_inlet")

gmsh.model.addPhysicalGroup(2, outlet_thruster_surfaces, 4)
gmsh.model.setPhysicalName(2, 4, "Thruster_outlet")

gmsh.model.addPhysicalGroup(2, cylinder_wall_surfaces, 5)
gmsh.model.setPhysicalName(2, 5, "Walls")

gmsh.model.addPhysicalGroup(2, outlet_plume_surfaces, 6)
gmsh.model.setPhysicalName(2, 6, "Plume_outlet")

# Define parámetros de discretización (tamaño mínimo y máximo de los elementos)
gmsh.option.setNumber("Mesh.MeshSizeMin", R_big / 10)
gmsh.option.setNumber("Mesh.MeshSizeMax", R_big / 5)

# Genera la malla 3D
gmsh.model.mesh.generate(3)

# Exporta la malla en formato .msh (de Gmsh)
gmsh.write("SimulationZone.msh")

# Crea y exporta la malla en formato XDMF (compatible con FEniCSx)
create_mesh(MPI.COMM_WORLD, gmsh.model, "SPT100_Simulation_Zone", "SimulationZone.xdmf", "w")

# Finaliza la sesión de Gmsh
gmsh.finalize()

