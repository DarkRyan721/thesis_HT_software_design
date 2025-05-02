import numpy as np
import pyvista as pv
import time

#_____________________________________________________________________________________________________
#           1] Cargar posiciones de las particulas en el tiempo

all_positions = np.load("data_files/particle_simulation.npy", mmap_mode="r")
num_frames, num_particles, _ = all_positions.shape

# print(all_positions[0])

# all_positions = all_positions[:, :, :3]

#_____________________________________________________________________________________________________
#           2] Funcion para detener la simulacion

window_closed = False # Variable global que detecta si la ventana sigue abierta
pausado = {"valor": False} # Variable global para el estado de pausa

def on_close():
    global window_closed
    window_closed = True

def pausar_simulacion():
    pausado["valor"] = not pausado["valor"]
    estado = "Pausada" if pausado["valor"] else "Ranudada"
    print(f"\n⏸️  Simulación: {estado}")

#_____________________________________________________________________________________________________
#           3] Funcion para la creacion de solidos/geometrias del motor

# Parametros de geometrias
L = 0.02
Rext = 0.05
Rint = 0.023
Rsol_ext = Rint/2
ancho_plano = (Rext*2)+(Rint)
espesor_plano = 0.001

def Geometries_creation():
    centro = (0, 0, (L) / 2)
    centro_solenoid = (ancho_plano/2)-Rsol_ext
    direccion = (0, 0, 1)

    # Cilindro hueco:
    cilindro_ext = pv.Cylinder(center=centro, direction=direccion, radius=Rext, height=L, resolution=200).triangulate()
    cilindro_int = pv.Cylinder(center=centro, direction=direccion, radius=Rint, height=L+0.001, resolution=200).triangulate()
    cilindro_tapa = pv.Cylinder(center=centro, direction=direccion, radius=Rint-1e-4, height=L+0.001, resolution=200).triangulate()

    # Boolean con limpieza
    cilindro_hueco = cilindro_ext.boolean_difference(cilindro_int).clean()

    # Clip más suave
    cilindro = cilindro_hueco.clip(normal="z", origin=(centro[0], centro[1], L - 1e-6), invert=True)

    # Plano solido:
    plano_solid = pv.Cube(center=(0, 0, 0), x_length=ancho_plano, y_length=ancho_plano, z_length=espesor_plano).triangulate()

    # Plano hueco:
    plano_hueco_aux = pv.Cube(center=(0, 0, L), x_length=ancho_plano, y_length=ancho_plano, z_length=espesor_plano).triangulate()
    cilindro_corte = pv.Cylinder(center=(0, 0, L), direction=direccion,radius=Rext,height=10,resolution=100).triangulate()

    plano_hueco = plano_hueco_aux.boolean_difference(cilindro_corte)

    cilindro_1 = pv.Cylinder(center=(centro_solenoid, centro_solenoid, (L)/2), direction=(0, 0, 1), radius=Rsol_ext, height=L, resolution=50)
    cilindro_2 = pv.Cylinder(center=(centro_solenoid, -centro_solenoid, (L)/2), direction=(0, 0, 1), radius=Rsol_ext, height=L, resolution=50)
    cilindro_3 = pv.Cylinder(center=(-centro_solenoid, centro_solenoid, (L)/2), direction=(0, 0, 1), radius=Rsol_ext, height=L, resolution=50)
    cilindro_4 = pv.Cylinder(center=(-centro_solenoid, -centro_solenoid, (L)/2), direction=(0, 0, 1), radius=Rsol_ext, height=L, resolution=50)

    plotter = pv.Plotter()
    plotter.set_background("black")
    plotter.add_mesh(cilindro, color="#656565", opacity=1, show_edges=False, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.2)
    plotter.add_mesh(plano_solid, color="gray", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
    plotter.add_mesh(plano_hueco, color="gray", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
    plotter.add_mesh(cilindro_tapa, color="gray", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
    plotter.add_mesh(cilindro_1, color="#CD7F32", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
    plotter.add_mesh(cilindro_2, color="#CD7F32", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
    plotter.add_mesh(cilindro_3, color="#CD7F32", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
    plotter.add_mesh(cilindro_4, color="#CD7F32", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)

    return plotter


plotter = Geometries_creation()
# plotter = pv.Plotter()
# plotter.set_background("black")

#_____________________________________________________________________________________________________
#           4] Configuracion de la camara

# Configurar cámara
plotter.camera_position = [(-5*Rext, 2.5*Rext, 5*Rext), (0, 0, 0), (0, 1, 0)]
plotter.camera.view_angle = 60  # Gran angular

# Iluminacion adicional
light = pv.Light(position=(-5*Rext, -5*Rext, -7*Rext), focal_point=(0, 0, 0), intensity=1.5)
plotter.add_light(light)

# Crear arreglo de partículas inicial SEPARADO
frame_data = all_positions[0]

mask_ions = frame_data[:, 3] == 1
mask_neutrals = frame_data[:, 3] == 0

ions_points = frame_data[mask_ions, :3]
neutrals_points = frame_data[mask_neutrals, :3]

ions = pv.PolyData(ions_points)
neutrals = pv.PolyData(neutrals_points)

# Añadir iones y neutros al plotter con diferentes colores
if ions.n_points > 0:
    ion_actor = plotter.add_mesh(ions, color='deepskyblue', point_size=1.0, render_points_as_spheres=True, lighting=True, specular=0.9, diffuse=1, ambient=0.3)
else:
    ion_actor = None  # No hay iones en el primer frame

if neutrals.n_points > 0:
    neutral_actor = plotter.add_mesh(neutrals, color='red', point_size=1.5, render_points_as_spheres=True, lighting=True, specular=0.9, diffuse=1, ambient=0.3)
else:
    neutral_actor = None  # No hay neutrales en el primer frame


plotter.add_text("\nHall Effect Thruster", position="upper_edge", color='white')

# Callback de cierre de ventana
plotter.iren.add_observer("ExitEvent", lambda *_: on_close())

x_line = pv.Line(pointa=(-100, 0, 0), pointb=(100, 0, 0))
y_line = pv.Line(pointa=(0, -100, 0), pointb=(0, 100, 0))
z_line = pv.Line(pointa=(0, 0, -100), pointb=(0, 0, 100))

# Agregamos las mallas de los ejes
plotter.add_mesh(x_line, color='red', line_width=3)
plotter.add_mesh(y_line, color='green', line_width=3)
plotter.add_mesh(z_line, color='blue', line_width=3)

# Mostrar ventana interactiva
plotter.show(auto_close=False, interactive_update=True)
plotter.add_key_event("space", pausar_simulacion)

#_____________________________________________________________________________________________________
#           5] Animacion

max_particles = num_particles  # Tamaño máximo del buffer
# Crear buffers para iones y neutros
buffer_ions = np.full((max_particles, 3), np.nan, dtype=np.float32)
buffer_neutrals = np.full((max_particles, 3), np.nan, dtype=np.float32)


for frame in range(num_frames):
    if window_closed:
        print("\n")
        break

    while pausado["valor"]:
        plotter.update()
        time.sleep(0.001)
        frame -= 1

    frame_data = all_positions[frame]

    mask_ions = frame_data[:, 3] == 1
    mask_neutrals = frame_data[:, 3] == 0
    ions_points = frame_data[mask_ions, :3]
    neutrals_points = frame_data[mask_neutrals, :3]

    # Update ions
    num_ions = min(len(ions_points), max_particles)

    if ion_actor is not None:
        if num_ions > 0:
            buffer_ions[:num_ions] = ions_points[:num_ions]
            ion_actor.SetVisibility(True)
            ions.points = buffer_ions
            ion_actor.mapper.dataset.points = ions.points
        else:
            ion_actor.SetVisibility(False)

        #ion_actor.SetVisibility(False)

    # Update neutrals
    num_neutrals = min(len(neutrals_points), max_particles)

    if neutral_actor is not None:
        if num_neutrals > 0:
            buffer_neutrals[:num_neutrals] = neutrals_points[:num_neutrals]
            neutral_actor.SetVisibility(True)
            neutrals.points = buffer_neutrals
            neutral_actor.mapper.dataset.points = neutrals.points
        else:
            neutral_actor.SetVisibility(False)

        neutral_actor.SetVisibility(False) 

    plotter.update()
    time.sleep(1 / 20)
    print(f"\rFrame: {frame + 1}/{num_frames} | Iones: {num_ions} | Neutros: {num_neutrals}", end='', flush=True)

print("\n")
