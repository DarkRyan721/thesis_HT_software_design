import numpy as np
import random
import pyvista as pv
import time



pv.global_theme.allow_empty_mesh = True

#_____________________________________________________________________________________________________
#           1] Cargar posiciones de las partículas en el tiempo

all_positions = np.load("data_files/particle_simulation.npy", mmap_mode="r")
num_frames, num_particles, _ = all_positions.shape

#_____________________________________________________________________________________________________
#           2] Funcion para detener la simulación

# Variable global que detecta si la ventana sigue abierta
window_closed = False

def on_close():
    global window_closed
    window_closed = True

#_____________________________________________________________________________________________________
#           3] Parámetros de ionización

ionization_threshold = 50  # Umbral de energía para ionizar (en eV)
ionization_probability = 0.05  # Probabilidad de ionización
energy_loss_per_ionization = 10  # Energía perdida por el electrón al ionizar (en eV)

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

#_____________________________________________________________________________________________________
#           4] Creación de geometrías del motor

def Geometries_creation():
    centro = (0, 0, 30)  # Centro del cilindro
    direccion = (0, 0, 1)

    # Cilindro hueco:
    cilindro_ext = pv.Cylinder(center=centro, direction=direccion, radius=42, height=60, resolution=50).triangulate()
    cilindro_int = pv.Cylinder(center=centro, direction=direccion, radius=18, height=60, resolution=50).triangulate()
    cilindro_hueco = cilindro_ext.boolean_difference(cilindro_int)

    cilindro = cilindro_hueco.clip(normal="z", origin=(0, 0, 59.999), invert=True)

    # Plano solido:
    plano_solid = pv.Cube(center=(0, 0, 0), x_length=110, y_length=110, z_length=2).triangulate()

    # Plano hueco:
    plano_hueco_aux = pv.Cube(center=(0, 0, 60), x_length=110, y_length=110, z_length=2).triangulate()
    cilindro_corte = pv.Cylinder(center=(0, 0, 60), direction=direccion, radius=42, height=10, resolution=100).triangulate()
    plano_hueco = plano_hueco_aux.boolean_difference(cilindro_corte)

    # Cilindros adicionales
    cilindro_1 = pv.Cylinder(center=(45, 45, 30), direction=(0, 0, 1), radius=10, height=60, resolution=50)
    cilindro_2 = pv.Cylinder(center=(-45, 45, 30), direction=(0, 0, 1), radius=10, height=60, resolution=50)
    cilindro_3 = pv.Cylinder(center=(45, -45, 30), direction=(0, 0, 1), radius=10, height=60, resolution=50)
    cilindro_4 = pv.Cylinder(center=(-45, -45, 30), direction=(0, 0, 1), radius=10, height=60, resolution=50)

    plotter = pv.Plotter()
    plotter.set_background("black")
    plotter.add_mesh(cilindro, color="#656565", opacity=1, show_edges=False, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.2)
    plotter.add_mesh(plano_solid, color="gray", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
    plotter.add_mesh(plano_hueco, color="gray", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
    plotter.add_mesh(cilindro_1, color="#CD7F32", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
    plotter.add_mesh(cilindro_2, color="#CD7F32", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
    plotter.add_mesh(cilindro_3, color="#CD7F32", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
    plotter.add_mesh(cilindro_4, color="#CD7F32", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)

    return plotter

# Crear Plotter
plotter = Geometries_creation()

#_____________________________________________________________________________________________________
#           5] Configuración de la cámara

center = np.array([0, 0, 90])  # Centro de la escena
plotter.camera_position = [(70, 70, 220), center, (0, 1, 0)]
plotter.camera.azimuth = 265
plotter.camera.elevation = 15
plotter.camera.view_angle = 115

# Iluminación adicional
light = pv.Light(position=(300, 300, -500), focal_point=(0, 0, 150), intensity=1.5)
plotter.add_light(light)

# Crear arreglo de partículas inicial
particles = pv.PolyData(all_positions[0])
particles['energy'] = np.random.uniform(0, 100, num_particles)  # Energías iniciales

# Añadir partículas al plotter
electron_actor = plotter.add_mesh(particles, color=electron_color, point_size=0.5, render_points_as_spheres=True, lighting=True, specular=0.9, diffuse=1, ambient=0.3)
ion_actor = plotter.add_mesh(pv.PolyData(), color=ion_color, point_size=0.5, render_points_as_spheres=True)

# Callback de cierre de ventana
plotter.iren.add_observer("ExitEvent", lambda *_: on_close())

# Ejes de referencia
x_line = pv.Line(pointa=(-100, 0, 0), pointb=(100, 0, 0))
y_line = pv.Line(pointa=(0, -100, 0), pointb=(0, 100, 0))
z_line = pv.Line(pointa=(0, 0, -100), pointb=(0, 0, 100))
plotter.add_mesh(x_line, color='red', line_width=3)
plotter.add_mesh(y_line, color='green', line_width=3)
plotter.add_mesh(z_line, color='blue', line_width=3)

# Mostrar ventana interactiva
plotter.show(auto_close=False, interactive_update=True)

#_____________________________________________________________________________________________________
#           6] Animación con ionización

for frame in range(num_frames):
    if window_closed:
        print("\n")
        break

    # Actualizar posiciones de las partículas
    particles.points = all_positions[frame]
    
    # Proceso de ionización
    new_ions_pos = []
    for i in range(particles.n_points):
        energy = particles['energy'][i]
        if ionize(energy):
            ion_pos, new_energy = create_ion(particles.points[i], energy)
            new_ions_pos.append(ion_pos)
            particles['energy'][i] = new_energy
    
    # Actualizar visualización
    electron_actor.mapper.dataset.points = particles.points
    if new_ions_pos:
        ion_actor.mapper.dataset.points = np.vstack([ion_actor.mapper.dataset.points, new_ions_pos])
    
    plotter.update()
    time.sleep(1/50)
    print(f"\rFrame: {frame + 1}/{num_frames}", end='', flush=True)