import os
import numpy as np
from vispy import app, scene
from vispy.geometry import create_cylinder
from vispy.scene import visuals
from vispy.visuals.filters import ShadingFilter
from vispy.geometry import MeshData

os.environ["VISPY_USE_APP"] = "glfw" # Backedn compatible

#_____________________________________________________________________________________________________
#                               Tapas de los cilindros

def caps_to_cylinder(cylinder, radius_top, radius_bottom, length):
    # Obtener vértices y caras del cilindro
    vertices = cylinder.get_vertices()
    faces = cylinder.get_faces()

    # Obtener el número de columnas (cols) del cilindro
    cols = int(len(vertices) / 20)  # Asumiendo rows=20 (ajusta según tu caso)

    # Generar vértices para las tapas
    theta = np.linspace(0, 2 * np.pi, cols, endpoint=False)

    # Tapa superior (usando radius_top)
    x_top = radius_top * np.cos(theta)  # Coordenadas X de la tapa superior
    y_top = radius_top * np.sin(theta)  # Coordenadas Y de la tapa superior
    z_top = np.full_like(x_top, length)  # Tapa superior (z = length)

    # Tapa inferior (usando radius_bottom)
    x_bottom = radius_bottom * np.cos(theta)  # Coordenadas X de la tapa inferior
    y_bottom = radius_bottom * np.sin(theta)  # Coordenadas Y de la tapa inferior
    z_bottom = np.zeros_like(x_bottom)        # Tapa inferior (z = 0)

    # Vértices de las tapas
    vertices_top = np.column_stack((x_top, y_top, z_top))
    vertices_bottom = np.column_stack((x_bottom, y_bottom, z_bottom))

    # Combinar todos los vértices
    vertices = np.vstack((vertices, vertices_top, vertices_bottom))

    # Generar caras para las tapas
    num_original_vertices = len(vertices) - 2 * cols  # Vértices originales del cilindro
    center_top = len(vertices) - 2 * cols  # Índice del centro de la tapa superior
    center_bottom = len(vertices) - cols   # Índice del centro de la tapa inferior

    # Tapa superior
    for j in range(cols):
        a = num_original_vertices + j
        b = num_original_vertices + (j + 1) % cols
        faces = np.vstack((faces, [a, b, center_top]))

    # Tapa inferior
    for j in range(cols):
        a = num_original_vertices + cols + j
        b = num_original_vertices + cols + (j + 1) % cols
        faces = np.vstack((faces, [a, center_bottom, b]))

    # Crear un nuevo MeshData con las tapas añadidas
    return MeshData(vertices=vertices, faces=faces)

# Cargar los datos de la simulación previamente guardados
all_positions = np.load("particle_simulation.npy", mmap_mode="r")
num_frames, num_particles, _ = all_positions.shape

# Crear la ventana Vispy
canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', size=(800, 600), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=60, elevation=30, azimuth=30)

# Establecer los límites correctamente
view.camera.center = (60, 60, 110)  # Centro de la escena
view.camera.scale_factor = 260  # Ajuste del zoom

# Crear el objeto de dispersión (scatter) para las partículas
scatter = scene.visuals.Markers(parent=view.scene)
scatter.set_gl_state('translucent', depth_test=True)
# -- Colores de las partículas --
# Puedes usar RGBA o nombres de color. Aquí un tono celeste/azulado.
plasma_color = (0.0, 0.8, 1.0, 1.0)  # R, G, B, A  (un celeste brillante)
scatter.set_data(all_positions[0],
                 edge_color=plasma_color,      # Borde del mismo color
                 face_color=plasma_color,
                 size=3.0)  # Ajuste de tamaño

# Prealocar memoria para evitar overhead en la actualización de datos
positions_buffer = np.empty_like(all_positions[0])

radius_outer = 42
radius_inner = 18
length = 60  # De 120 a 180 en Z
center_xy = (60, 60)
z_start = 120  # Posición base del cilindro en Z

# Crear cilindro externo
cylinder_outer = create_cylinder(rows=20, cols=100, radius=(radius_outer, radius_outer), length=length)

# Aplicar un material metálico al cilindro externo
mesh_outer = visuals.Mesh(meshdata=cylinder_outer, color=(0.7, 0.7, 0.7, 1))  # Gris claro
shading_filter_outer = ShadingFilter(shininess=100, specular_light=(1, 1, 1, 1))  # Reflejos metálicos
mesh_outer.attach(shading_filter_outer)
mesh_outer.transform = scene.transforms.STTransform(translate=(center_xy[0], center_xy[1], z_start))

# Crear el cilindro interno
cylinder_inner = create_cylinder(rows=20, cols=100, radius=(radius_inner, radius_inner), length=length)

# Añadir tapas al cilindro interno
cylinder_inner_with_caps = caps_to_cylinder(cylinder_inner, radius_top=radius_outer, radius_bottom=radius_inner, length=length)

# Crear la malla con las tapas
mesh_inner = visuals.Mesh(meshdata=cylinder_inner_with_caps, color=(0.3, 0.3, 0.3, 1))
shading_filter_inner = ShadingFilter(shininess=100, specular_light=(1, 1, 1, 1))
mesh_inner.attach(shading_filter_inner)
mesh_inner.transform = scene.transforms.STTransform(translate=(center_xy[0], center_xy[1], z_start))

# Agregar el cilindro interno a la vista
view.add(mesh_inner)

# Agregar cilindros a la vista
view.add(mesh_outer)

# Contador de frames
frame_idx = 0

# Función de actualización de la animación
def update(event):
    global frame_idx, positions_buffer

    if frame_idx < num_frames:
        # Copiamos datos del frame actual
        np.copyto(positions_buffer, all_positions[frame_idx])
        scatter.set_data(positions_buffer,
                         size=1.0,
                         face_color=plasma_color,
                         edge_color=plasma_color)
        
        print(f"Frame: {frame_idx+1}/{num_frames}")
        frame_idx += 1
    else:
        # Detenemos el temporizador al llegar al último frame
        timer.stop()
        print("¡Simulación finalizada!")

# Configurar un temporizador para actualizar la animación
timer = app.Timer(interval=1 / 60, connect=update, start=True)  # 60 FPS para mayor fluidez

# Mostrar la ventana correctamente
if __name__ == '__main__':
    canvas.show()
    app.run()