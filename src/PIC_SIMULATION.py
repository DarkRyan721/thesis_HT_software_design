import numpy as np
import pyvista as pv
import time

#_____________________________________________________________________________________________________
#           1] Cargar posiciones de las particulas en el tiempo

all_positions = np.load("particle_simulation.npy", mmap_mode="r")
num_frames, num_particles, _ = all_positions.shape

#_____________________________________________________________________________________________________
#           2] Funcion para detener la simulacion

# Variable global que detecta si la ventana sigue abierta
window_closed = False

def on_close():
    global window_closed
    window_closed = True

#_____________________________________________________________________________________________________
#           3] Funcion para la creacion de solidos/geometrias del motor

# Parametros de geometrias
centro_xy = (60, 60)
z_inicio = 120
z_final = 180
altura = 60
Rext = 42
Rint = 18
ancho_plano = 110
espesor_plano = 2

def Geometries_creation():
    centro = (centro_xy[0], centro_xy[1], (z_inicio + z_final) / 2)
    direccion = (0, 0, 1)

    # Cilindro hueco:
    cilindro_ext = pv.Cylinder(center=centro, direction=direccion, radius=Rext, height=altura, resolution=50).triangulate()
    cilindro_int = pv.Cylinder(center=centro, direction=direccion, radius=Rint, height=altura, resolution=50).triangulate()
    cilindro_hueco = cilindro_ext.boolean_difference(cilindro_int)

    cilindro = cilindro_hueco.clip(normal="z", origin=(centro_xy[0], centro[1], z_inicio), invert=False)

    # Plano solido:
    plano_solid = pv.Cube(center=(centro_xy[0], centro_xy[1], z_final), x_length=ancho_plano, y_length=ancho_plano, z_length=espesor_plano).triangulate()

    # Plano hueco:
    plano_hueco_aux = pv.Cube(center=(centro_xy[0], centro_xy[1], z_inicio), x_length=ancho_plano, y_length=ancho_plano, z_length=espesor_plano).triangulate()
    cilindro_corte = pv.Cylinder(center=(centro_xy[0], centro_xy[1], z_inicio), direction=direccion,radius=Rext,height=10,resolution=100).triangulate()

    plano_hueco = plano_hueco_aux.boolean_difference(cilindro_corte)

    cilindro_1 = pv.Cylinder(center=(centro_xy[0]+45, centro_xy[1]+45, (z_inicio + z_final) / 2), direction=(0, 0, 1), radius=10, height=60, resolution=50)
    cilindro_2 = pv.Cylinder(center=(centro_xy[0]-45, centro_xy[1]+45, (z_inicio + z_final) / 2), direction=(0, 0, 1), radius=10, height=60, resolution=50)
    cilindro_3 = pv.Cylinder(center=(centro_xy[0]+45, centro_xy[1]-45, (z_inicio + z_final) / 2), direction=(0, 0, 1), radius=10, height=60, resolution=50)
    cilindro_4 = pv.Cylinder(center=(centro_xy[0]-45, centro_xy[1]-45, (z_inicio + z_final) / 2), direction=(0, 0, 1), radius=10, height=60, resolution=50)

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
#           4] Configuracion de la camara

# Calcular centro
center = np.array([(20 + 120)/2, (20 + 120)/2, (120 + 180)/2])

# Configurar cámara
plotter.camera_position = [(70, 70, 350), center, (0, 1, 0)]
plotter.camera.azimuth = 225
plotter.camera.elevation = 30
plotter.camera.view_angle = 105

# Iluminacion adicional
light = pv.Light(position=(300, 300, 500), focal_point=(60, 60, 150), intensity=1.5)
plotter.add_light(light)

# Crear arreglo de particulas inicial
particles = pv.PolyData(all_positions[0])

# Añadir partículas al plotter
particle_actor = plotter.add_mesh(particles, color='#74faf2', point_size=1.5, render_points_as_spheres=True, lighting=True, specular=0.9, diffuse=1, ambient=0.3)
plotter.add_text("\nHall Effect Thruster", position="upper_edge", color='white')

# Callback de cierre de ventana
plotter.iren.add_observer("ExitEvent", lambda *_: on_close())

# Mostrar ventana interactiva
plotter.show(auto_close=False, interactive_update=True)

#_____________________________________________________________________________________________________
#           5] Animacion

for frame in range(num_frames):
    if window_closed:
        print("\n")
        break

    particles.points = all_positions[frame]
    particle_actor.mapper.dataset.points = particles.points
    plotter.update()
    time.sleep(1/50)
    print(f"\rFrame: {frame + 1}/{num_frames}", end='', flush=True)