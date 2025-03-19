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
     existan y sean accesibles en el directorio data_files.
  2. Ejecutar con: python3 <nombre_de_este_script>.py

--------------------------------------------------------------------------------
"""

import os
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # Habilita proyecciones 3D en matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors

# Importaciones de dolfinx y UFL
from dolfinx import fem, io, mesh
import ufl
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

# Importaciones de petsc4py
from petsc4py import PETSc

#Para visualizacion
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as colors
import matplotlib.cm as cm


def solve_potential(domain, facet_tags, volt_tag=3, ground_tag=4, Volt=300):
    """
    Resuelve la ecuación de Laplace para el potencial eléctrico en el dominio dado.

    Parámetros:
    -----------
    domain : dolfinx.mesh.Mesh
        Malla cargada en FEniCSx.
    facet_tags : dolfinx.MeshTags
        Etiquetas de facetas (fronteras) para asignar condiciones de contorno.
    volt_tag : int
        Tag de la faceta donde se aplica un voltaje alto (Dirichlet).
    ground_tag : int
        Tag de la faceta puesta a tierra (0 V).
    Volt : float
        Valor del potencial en volt_tag.

    Retorna:
    --------
    phi_h : dolfinx.fem.Function
        Función solución para el potencial eléctrico.
    """
    # Espacio de funciones CG de orden 1 para el potencial
    V = fem.functionspace(domain, ("CG", 1))
    
    # Función que nos servirá para asignar los valores de la frontera
    u_bc = fem.Function(V)
    u_bc.x.array[:] = np.nan  # Inicialmente NaN para diferenciar zonas

    # Listado de condiciones de Dirichlet
    boundary_conditions = []

    # Asignamos un potencial alto (Volt) en el tag = volt_tag
    for tag, value in [(volt_tag, Volt), (ground_tag, 0.0)]:
        # Ubica los dofs pertenecientes a la faceta con etiqueta `tag`
        dofs = fem.locate_dofs_topological(
            V,
            domain.topology.dim - 1,
            facet_tags.indices[facet_tags.values == tag]
        )
        # Asigna el valor de potencial en esos dofs
        u_bc.x.array[dofs] = value
        
        # Crea la condición de Dirichlet y la agrega a la lista
        bc = fem.dirichletbc(u_bc, dofs)
        boundary_conditions.append(bc)

    # Se puede almacenar esta función de frontera para depuración o post-proceso
    with io.XDMFFile(domain.comm, "boundary_conditions.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(u_bc)

    # Definición débil de la ecuación de Laplace:
    # -∇·∇φ = 0  =>  ∫ (grad(phi)·grad(v)) dx = ∫ 0* v dx
    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = inner(grad(phi), grad(v)) * dx
    L = fem.Constant(domain, PETSc.ScalarType(0)) * v * dx

    # Configuración de la resolución del problema lineal
    problem = LinearProblem(
        a, L,
        bcs=boundary_conditions,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    phi_h = problem.solve()  # Resuelve la ecuación

    # Se puede almacenar la solución (potencial) para su visualización posterior
    with io.XDMFFile(domain.comm, "Laplace.xdmf", "w") as file:
        file.write_mesh(domain)
        file.write_function(phi_h)

    return phi_h


def compute_electric_field(domain, phi_h):
    """
    Calcula el campo eléctrico E = -∇φ y lo devuelve como una dolfinx.fem.Function.

    Parámetros:
    -----------
    domain : dolfinx.mesh.Mesh
        Malla cargada en FEniCSx.
    phi_h : dolfinx.fem.Function
        Función solución para el potencial eléctrico.

    Retorna:
    --------
    E_field : dolfinx.fem.Function
        Campo eléctrico, un vector en cada nodo.
    """
    # Espacio vectorial (dim igual a la dimensión espacial)
    V_vector = fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))

    # Definición de la expresión para E = -grad(phi_h)
    E_field = fem.Function(V_vector)
    E_field_expr = fem.Expression(-grad(phi_h), V_vector.element.interpolation_points())
    E_field.interpolate(E_field_expr)

    # Almacenamos el campo en un archivo XDMF
    with io.XDMFFile(domain.comm, "Electric_Field.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(E_field)

    return E_field

def main():
    """
    Función principal que realiza:
    1. Cambio de directorio a 'data_files' para leer y escribir archivos.
    2. Lectura de la malla y etiquetas desde SimulationZone.xdmf.
    3. Llamada a la rutina de resolución de Laplace para obtener el potencial.
    4. Cálculo del campo eléctrico como E = -∇φ.
    5. Visualización en 3D de los vectores de E.
    """
    # Cambiamos directorio para asegurarnos de que los archivos se lean/escriban en data_files
    os.chdir("data_files")

    # Se abre el archivo "SimulationZone.xdmf" para leer la malla y las etiquetas
    with io.XDMFFile(MPI.COMM_WORLD, "SimulationZone.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="SPT100_Simulation_Zone")

    # Crear la conectividad necesaria (faceta->celda)
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

    # Información básica de la malla
    print("Malla cargada exitosamente en FEniCSx")
    print(f"Dimensión de la malla: {domain.topology.dim}")
    print(f"Número de nodos: {domain.geometry.x.shape[0]}")

    # Se leen también las etiquetas de facetas y celdas
    with io.XDMFFile(MPI.COMM_WORLD, "SimulationZone.xdmf", "r") as xdmf:
        facet_tags = xdmf.read_meshtags(domain, name="SPT100_Simulation_Zone_facets")
        cell_tags = xdmf.read_meshtags(domain, name="SPT100_Simulation_Zone_cells")

    print("Etiquetas de frontera cargadas exitosamente.")
    print(f"Número de facetas etiquetadas: {len(facet_tags.values)}")
    print(f"Etiquetas disponibles: {set(facet_tags.values)}")

    # -------------------------------------------------------------------------
    # 1. Resolver el potencial (Laplace)
    # -------------------------------------------------------------------------
    #   - Por defecto, se aplican 10kV en las facetas con tag=3 (Gas_inlet)
    #   - 0V en las facetas con tag=4 (Thruster_outlet)
    # Puedes ajustar estos tags o valores según tu caso.
    phi_h = solve_potential(
        domain=domain,
        facet_tags=facet_tags,
        volt_tag=3,   # Gas_inlet
        ground_tag=4, # Thruster_outlet
        Volt=300
    )

    # -------------------------------------------------------------------------
    # 2. Calcular el campo eléctrico
    # -------------------------------------------------------------------------
    E_field = compute_electric_field(domain, phi_h)


    #3. Pasar el campo resultante a archivo tipo numpy

    # Obtiene las coordenadas (XYZ) de los nodos de la malla
    X = domain.geometry.x
    # El campo E_field.x.array contiene los valores del campo eléctrico en cada dof
    # Se reordena y se separa en Ex, Ey, Ez
    E_values = E_field.x.array.reshape(-1, domain.geometry.dim).T

    # Se concatena la información de coordenadas y campo eléctrico en un array (N x 6)
    # [x, y, z, Ex, Ey, Ez]
    E_np = np.hstack((X, E_values.T))
    np.save("Electric_Field_np.npy", E_np)


# ---------------------------------------------------------------------
# Ejecución como script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
