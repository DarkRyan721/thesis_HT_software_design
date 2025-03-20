import numpy as np
import random
import pyvista as pv
import time

# Habilitar mallas vacías
pv.global_theme.allow_empty_mesh = True

# Parámetros de ionización
ionization_threshold = 50  # Umbral de energía para ionizar (en eV)
ionization_probability = 0.05  # Probabilidad de ionización
energy_loss_per_ionization = 10  # Energía perdida por el electrón al ionizar (en eV)

# Cargar posiciones de los iones desde el archivo .npy
all_ion_positions = np.load("data_files/particle_simulation.npy", mmap_mode="r")
num_frames, num_ions, _ = all_ion_positions.shape  # Obtener dimensiones del archivo

# Generar dinámica de electrones (aleatoria, por simplicidad)
all_electron_positions = np.random.uniform(-1, 1, size=(num_frames, num_ions, 3))  # Posiciones aleatorias

# Colores para visualización
electron_color = "#74faf2"  # Color para electrones (azul claro)
ion_color = "#FF4500"  # Color para iones (naranja)

# Función para simular la ionización usando Montecarlo
def ionize(particle_energy):
    """Determina si una partícula se ioniza basándose en su energía."""
    if particle_energy > ionization_threshold:
        return random.random() < ionization_probability
    return False

# Función para crear nuevos iones
def create_ion(particle_position, particle_energy):
    """Crea un nuevo ion y actualiza la energía del electrón."""
    new_ion_position = particle_position + np.random.uniform(-0.1, 0.1, size=3)  # Posición aleatoria cercana
    new_ion_energy = 0  # Los iones no tienen energía cinética
    return new_ion_position, particle_energy - energy_loss_per_ionization  # Electrón pierde energía

# Configurar el plotter de PyVista
plotter = pv.Plotter()
plotter.set_background("black")  # Fondo negro

# Inicializar partículas (electrones e iones)
electrons = pv.PolyData(all_electron_positions[0])  # Cargar posiciones iniciales de electrones
electrons['energy'] = np.random.uniform(0, 100, num_ions)  # Asignar energías aleatorias (0-100 eV)

ions = pv.PolyData(all_ion_positions[0])  # Cargar posiciones iniciales de iones

# Crear actores para visualización
electron_actor = plotter.add_mesh(electrons, color=electron_color, point_size=5, render_points_as_spheres=True)
ion_actor = plotter.add_mesh(ions, color=ion_color, point_size=5, render_points_as_spheres=True)

# Mostrar la ventana de visualización
plotter.show(auto_close=False, interactive_update=True)

# Bucle principal de la simulación
for frame in range(num_frames):
    # Actualizar posiciones de los electrones e iones
    electrons.points = all_electron_positions[frame]
    ions.points = all_ion_positions[frame]
    
    # Proceso de ionización (solo para electrones)
    new_ions_pos = []
    for i in range(electrons.n_points):
        energy = electrons['energy'][i]  # Obtener energía del electrón
        if ionize(energy):  # Verificar si se ioniza
            # Crear un nuevo ion y actualizar la energía del electrón
            ion_pos, new_energy = create_ion(electrons.points[i], energy)
            new_ions_pos.append(ion_pos)  # Añadir nuevo ion
            electrons['energy'][i] = new_energy  # Actualizar energía del electrón
    
    # Actualizar posiciones de los iones (añadir nuevos iones)
    if new_ions_pos:
        ions.points = np.vstack([ions.points, new_ions_pos])
    
    # Actualizar visualización
    electron_actor.mapper.dataset.points = electrons.points  # Actualizar electrones
    ion_actor.mapper.dataset.points = ions.points  # Actualizar iones
    plotter.update()  # Refrescar la visualización
    
    # Controlar la velocidad de la simulación
    time.sleep(1/30)  # 30 frames por segundo (ajusta según sea necesario)

# Cerrar el plotter al finalizar
plotter.close()