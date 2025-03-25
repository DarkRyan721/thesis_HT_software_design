import numpy as np
import plotly.graph_objects as go

# Cargar datos
E_np = np.load('data_files/Electric_Field_np.npy')

# Filtrar datos con Z ≤ 0.03 (coordenada original)
mask = E_np[:, 2] <= 0.03
filtered_data = E_np[mask]

# Extraer coordenadas y componentes (rotadas)
x_rot = filtered_data[:, 2]  # Eje X visual = Z original
y_rot = filtered_data[:, 0]  # Eje Y visual = X original
z_rot = filtered_data[:, 1]  # Eje Z visual = Y original

# Calcular magnitud del campo
magnitude = np.sqrt(filtered_data[:, 3]**2 + filtered_data[:, 4]**2 + filtered_data[:, 5]**2)

# 1. Filtrar puntos en Z original = 0 (con tolerancia)
epsilon = 1e-5
mask_z0 = np.abs(filtered_data[:, 2]) < epsilon  # Z original ≈ 0
points_z0 = filtered_data[mask_z0]

# 2. Generar vectores y flechas
segments = []
arrow_positions = []
arrow_directions = []
colors = []

for point in points_z0:
    # Coordenadas originales del punto
    x_orig = point[0]  # X original (será Y visual)
    y_orig = point[1]  # Y original (será Z visual)
    
    # Punto inicial y final (rotados)
    start = [0, x_orig, y_orig]          # Z=0
    end = [0.01, x_orig, y_orig]         # Z=0.01
    
    # Línea
    segments.extend([start, end, [None]*3])
    
    # Flecha (posición final y dirección)
    arrow_positions.append(end)
    arrow_directions.append([0.002, 0, 0])  # Dirección X visual (Z original)
    
    # Color basado en magnitud
    mag = np.sqrt(point[3]**2 + point[4]**2 + point[5]**2)
    colors.append(mag)

# Convertir a arrays numpy
segments = np.array(segments)
x_lines, y_lines, z_lines = segments.T
arrow_positions = np.array(arrow_positions)
arrow_directions = np.array(arrow_directions)

# Crear figura
fig = go.Figure()

# Añadir puntos del campo eléctrico
fig.add_trace(go.Scatter3d(
    x=x_rot,
    y=y_rot,
    z=z_rot,
    mode='markers',
    marker=dict(
        size=1,
        color=magnitude,
        colorscale='Plasma',
        opacity=0.7,
        colorbar=dict(title='|E| (N/C)')
)))

# Añadir líneas de vectores
fig.add_trace(go.Scatter3d(
    x=x_lines,
    y=y_lines,
    z=z_lines,
    mode='lines',
    line=dict(
        color="yellow",
        width=7,
        cmin=np.min(magnitude),
        cmax=np.max(magnitude)
    ),
    opacity=0.4,
    name='Vectores'
))

# Añadir flechas (conos)
fig.add_trace(go.Cone(
    x=arrow_positions[:, 0],  # Posiciones X (Z original)
    y=arrow_positions[:, 1],  # Posiciones Y (X original)
    z=arrow_positions[:, 2],  # Posiciones Z (Y original)
    u=arrow_directions[:, 0],  # Dirección X (Z original)
    v=arrow_directions[:, 1],  # Dirección Y (X original)
    w=arrow_directions[:, 2],  # Dirección Z (Y original)
    colorscale=[[0, 'yellow'], [1, 'yellow']],
    #colorscale='Plasma',
    sizemode="absolute",
    sizeref=0.001,  # Ajustar tamaño de flechas
    anchor="tip",
    showscale=False
))

# Configuración de ejes y cámara
fig.update_layout(
    title='Campo Eléctrico con Vectores en Z=0 (Z ≤ 0.03 m)',
    scene=dict(
        xaxis=dict(title='Z [m]', range=[0, 0.03]),  # Eje X visual = Z original
        yaxis=dict(title='X [m]', range=[-0.1, 0.1]),  # Eje Y visual = X original
        zaxis=dict(title='Y [m]', range=[-0.1, 0.1]),  # Eje Z visual = Y original
        camera=dict(
            eye=dict(x=0.5, y=1.5, z=0.5),  # Vista oblicua
            up=dict(x=0, y=0, z=1)
        )
    ),
    template='plotly_dark'
)

fig.show()