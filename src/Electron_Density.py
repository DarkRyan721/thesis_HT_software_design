import numpy as np
from dolfinx import fem, io
from ufl import SpatialCoordinate, sqrt
import pyvista as pv
from scipy.stats import gamma
from tqdm import tqdm
import os
from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def gamma_custom(z_val, k, theta, z_min):
    """
    gamma_custom:

    Funcion que recibe los principales parametros de la funcion de gamma y retorna esta misma.
    """

    return gamma.pdf(z_val, a=k, scale=theta, loc=z_min)

def generate_density(domain, r0=0.04, sigma_r=0.007, A=1.2e15, z_min=-0.004, k=2, theta=0.012):
    """
    generate_density:

    Funcion que recibe el dominio de la malla y los multiples parametros necesarios para crear
    la distribucion espacial de la densidad electrónica n_e(r,z) combinando una funcion gamma
    axial(eje Z) con una funcion gaussiana para su parte radial. Retornando la funcion de
    densidad electornica en la malla en formato numpy.

    domain -> variable que contiene el dominio/malla de la simulacion
    r0 -> es la posicion radial en la cual se encuentra el pico de la densidad electronica
    sigma_r -> variable que define la dispersion(desviacion) de la densidad radialmente
    A -> es la amplitud o valor maximo que se encontrara en la densidad de carga
    z_min -> variable que ayuda a ubicar el punto pico de la densidad en el eje axial(Z)
    k -> parametro que controla la caida de densidad a lo largo del eje axial(Z)
    theta -> controla la longitud de la extension de la funcion gamma
    """

    #___________________________________________________________________________________________
    #       Creacion de la barra de progreso para la informacion del proceso

    pbar = tqdm(total=4, desc="Densidad de carga")

    #___________________________________________________________________________________________
    #       Creacion del espacio de funciones usando el dominio de la malla

    V = fem.functionspace(domain, ("CG", 1))

    pbar.update(1)

    #___________________________________________________________________________________________
    #       Obtencion de los puntos interpolando el espacio de funciones V

    x_points = V.element.interpolation_points()
    ne_vals = np.zeros(len(x_points))

    pbar.update(1)

    #___________________________________________________________________________________________
    #       Calculo de la densidad electronica para cada punto en el espacio

    for i, pt in enumerate(x_points):
        r_val = np.sqrt(pt[0]**2 + pt[1]**2)
        z_val = pt[2]
        val_r = np.exp(-((r_val - r0)**2) / (2 * sigma_r**2))
        val_z = gamma_custom(z_val, k, theta, z_min)
        ne_vals[i] = A * val_r * val_z

    pbar.update(1)

    #___________________________________________________________________________________________
    #       Interpolacion de los valores de densidad de carga a la malla(dominio)

    n_e = fem.Function(V)

    n_e.interpolate(lambda x: np.array([
        A * np.exp(-((np.sqrt(xi[0]**2 + xi[1]**2) - r0)**2) / (2 * sigma_r**2)) *
        gamma_custom(xi[2], k, theta, z_min)
        for xi in x.T]).reshape((1, -1)))
    
    pbar.update(1)
    pbar.close()

    return n_e

    #___________________________________________________________________________________________

def save_density(n0, filename="density_n0.npy"):
    """
    save_density:

    Guarda la densidad de carga en un archivo .npy
    """
    #___________________________________________________________________________________________

    n0_array = n0.x.array
    np.save(filename, n0_array)

    #___________________________________________________________________________________________

def plot_density_XY(r0=0.04, sigma_r=0.007, A=1.2e15, z_min=-0.004, k=2, theta=0.012, z_plane=0.01, x_range=(-0.1, 0.1), y_range=(-0.1, 0.1), resolution=5000, Rin=0.028, Rex=0.05):

    #___________________________________________________________________________________________
    #       Crear malla 2D

    x_grid, y_grid = np.mgrid[x_range[0]:x_range[1]:resolution*1j, y_range[0]:y_range[1]:resolution*1j]
    r_vals = np.sqrt(x_grid**2 + y_grid**2)
    z_vals = np.full_like(x_grid, z_plane)

    #___________________________________________________________________________________________
    #     Calcular densidad en el plano z_plane 

    density_vals = A * np.exp(-((r_vals - r0)**2) / (2 * sigma_r**2)) * gamma_custom(z_vals, k, theta, z_min)

    global_max = np.max(density_vals)
    global_min = global_max / 10

    density_vals[density_vals < 1] = 0


    #___________________________________________________________________________________________
    #       definicion de máscara para el anillo: Rin < R < Rex

    mask = (r_vals >= Rin) & (r_vals <= Rex)
    masked_density = np.ma.masked_where(~mask, density_vals)
    
    #___________________________________________________________________________________________
    #       Asignar valores fuera del anillo como cero

    masked_density = np.ma.filled(masked_density, 0)

    #___________________________________________________________________________________________
    #       Creacion y configuracion de la figura para el plot
    
    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title("Plano de Densidad Electrónica")

    plt.gca().set_facecolor('black') 

    ticks = np.logspace(np.log10(global_min), np.log10(global_max), num=5)
    tick_labels = [f'{tick:.1e}' for tick in ticks]

    norm = mcolors.LogNorm(vmin=global_min, vmax=global_max)

    img = plt.imshow(masked_density, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), origin='lower', cmap='inferno', aspect='auto', norm=norm)

    cbar = plt.colorbar(img, label='ne (m⁻³)')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)

    plt.title(f"Densidad de electrones en z = {z_plane}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()

    #___________________________________________________________________________________________

def plot_density_ZX(r0=0.04, sigma_r=0.007, A=1.2e15, z_min=-0.004, k=2, theta=0.012, y_plane=0.0, x_range=(0.0, 0.1), z_range=(0, 0.2), resolution=5000, global_max=3.67e16, Rin=0.028, Rex=0.05):
    
    #___________________________________________________________________________________________
    #       Crear malla (Z vs X)

    z_grid, x_grid = np.mgrid[z_range[0]:z_range[1]:resolution*1j, x_range[0]:x_range[1]:resolution*1j]
    y_vals = np.full_like(x_grid, y_plane)
    r_vals = np.sqrt(x_grid**2 + y_vals**2)

    #___________________________________________________________________________________________
    #       Calcular densidad

    density_vals = A * np.exp(-((r_vals - r0)**2) / (2 * sigma_r**2)) * gamma_custom(z_grid, k, theta, z_min)

    global_max = np.max(density_vals)
    global_min = global_max / 1e6

    density_vals[density_vals < 1] = 0

    #___________________________________________________________________________________________
    #       Máscara condicional para correcta impresion

    mask_condition = (z_grid <= 0.02) & ((np.abs(x_grid) < Rin) | (np.abs(x_grid) > Rex))
    masked_density = np.where(mask_condition, 0.0, density_vals)
    masked_density[masked_density < 1] = 0 

    #___________________________________________________________________________________________
    #       Creacion y configuracion del gráfico

    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title("Plano de Densidad Electrónica")
    plt.gca().set_facecolor('black')

    ticks = np.logspace(np.log10(global_min), np.log10(global_max), num=5)
    tick_labels = [f'{tick:.1e}' for tick in ticks]

    norm = mcolors.LogNorm(vmin=global_min, vmax=global_max)

    cmap = plt.get_cmap('inferno').copy()
    cmap.set_under('black')

    img = plt.imshow(masked_density.T, extent=(z_range[0], z_range[1], x_range[0], x_range[1]), origin='lower', cmap=cmap, aspect='auto', norm=norm)

    cbar = plt.colorbar(img, label='ne (m⁻³)')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)

    plt.title(f"Densidad de electrones en Y = {y_plane}")
    plt.xlabel("Z (m)")
    plt.ylabel("X (m)")
    plt.show()

    #___________________________________________________________________________________________

def plot_density(title="Densidad Electronica", bool_3D = True, bool_XY_Plane = False, bool_XZ_plane = False):
    """
        plot_density:

        Funcion que plotea la densidad de electrones en diferente opciones como 3D y 2D en XY(radial) o
        XZ(axial).

        title -> Titulo del plot
        bool_3D -> Plotea la densidad en 3 dimensiones
        bool_XY_Plane -> Plotea la densidad en el plano XY o plano radial
        bool_XZ_Plane -> Plotea la densidad en el plano XZ o plano axial teoricamente equivalente al plano YZ
    """

    if bool_3D == True:
        #___________________________________________________________________________________________
        #       Cargando los datos espaciales y de densidad

        E_np = np.load('Electric_Field_np.npy')
        points = E_np[:, :3]

        n0 = np.load('density_end.npy')
        n0_log = np.log10(n0)

        #___________________________________________________________________________________________
        #       Creando el objeto plot

        mesh = pv.PolyData(points)

        max_value = np.max(n0_log[n0_log != np.log10(1e-100)])

        #___________________________________________________________________________________________
        #       Definir límites manuales para el colormap
        
        log_min = max_value - 3.0
        log_max = max_value

        #___________________________________________________________________________________________
        #       Configuracion del plot

        mesh["n0_log"] = n0_log

        plotter = pv.Plotter()
        plotter.add_axes(color="white")
        plotter.set_background("black")

        plotter.add_mesh(
            mesh, 
            scalars="n0_log", 
            cmap="plasma", 
            clim=[log_min, log_max],
            point_size=2,
            render_points_as_spheres=True,
            scalar_bar_args={
                'title': "ne [m⁻³] (log₁₀)\n",
                'color': 'white',
                'fmt': "%.1f",
            }
        )

        plotter.camera_position = [
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0), 
            (0.0, -1.0, 0.0)
        ]

        plotter.add_text("Distribución de Densidad Electrónica", position='upper_edge', font_size=12, color='white')

        plotter.show()

        #___________________________________________________________________________________________
    elif bool_XY_Plane == True:
        plot_density_XY()
    elif bool_XZ_plane == True:
        plot_density_ZX()
    else:
        print("Ningun plot seleccionado")

if __name__ == "__main__":
    os.chdir("data_files")

    # EXAMPLE FOR GUI:

    """
        * Entradas que debe brindar el usuario para la densidad de electrones:

        NINGUNA

        * Valores que cambian segun el caso(NO SE LES PIDE AL USUARIO):

        r0 -> sera el valor de (Rin + Rex)/2
        z_min -> debera depender de la profundiad, Aun no hay formula
        theta -> debera depender de la profundidad. Aun no hay formula
        sigma_r -> debera depender del r0. Aun no hay formula
        A -> debera depender de el rendimiento de la simulacion. No existe formula
    """

    #1. Cargar la malla
    with io.XDMFFile(MPI.COMM_WORLD, "SimulationZone.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="SPT100_Simulation_Zone")

    # 2. Generar la densidad de electrones inicial
    n0 = generate_density(domain)

    # 3. Guardar la densidad de electrones
    save_density(n0)

    # 3. plot opcional de la densidad de electrones
    plot_density(bool_3D=True, bool_XY_Plane=False, bool_XZ_plane=False) #Tiene 3 tipos de plots(3D, Plano XY, Plano ZX)