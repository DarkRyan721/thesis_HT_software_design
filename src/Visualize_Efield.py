import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Cargar datos
E_np = np.load('data_files/Electric_Field_np.npy')

# Filtrar datos con E > 0 (opcional, ajusta según necesidad)
mask = np.any(E_np[:, 3:] > 0, axis=1)
filtered_data = E_np[mask]

# Extraer coordenadas XYZ y componentes Ex, Ey, Ez
x = filtered_data[:, 0]   # Coordenada X
y = filtered_data[:, 1]   # Coordenada Y
z = filtered_data[:, 2]   # Coordenada Z
Ex = filtered_data[:, 3]  # Componente Ex
Ey = filtered_data[:, 4]  # Componente Ey
Ez = filtered_data[:, 5]  # Componente Ez

# Calcular magnitud del campo eléctrico para el mapa de colores
magnitude = np.sqrt(Ex**2 + Ey**2 + Ez**2)

# Crear figura 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Configurar mapa de colores (normalizado)
norm = plt.Normalize(vmin=np.min(magnitude), vmax=np.max(magnitude))
colors = cm.plasma(norm(magnitude))  # Paleta "plasma"

# Graficar vectores 3D con colores según la magnitud
quiver = ax.quiver(
    x, y, z,          # Posiciones (X, Y, Z)
    Ex, Ey, Ez,       # Componentes del campo (Ex, Ey, Ez)
    color=colors,      # Colores basados en la magnitud
    length=0.1,       # Longitud base de las flechas (ajusta según tus datos)
    arrow_length_ratio=0.5,  # Proporción cabeza/flecha
    linewidth=0.5,    # Grosor de las flechas
    alpha=0.8         # Transparencia
)

# Añadir barra de colores
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.plasma), ax=ax, shrink=0.7)
cbar.set_label('Magnitud del Campo Eléctrico (N/C)', fontsize=12)

# Añadir etiquetas y estilo profesional
ax.set_xlabel('X [m]', fontsize=12, labelpad=10)
ax.set_ylabel('Y [m]', fontsize=12, labelpad=10)
ax.set_zlabel('Z [m]', fontsize=12, labelpad=10)
ax.set_title('Campo Eléctrico 3D con Vectores y Mapa de Colores', fontsize=14, fontweight='bold', pad=20)

# Ajustar perspectiva (elevación y azimut)
ax.view_init(elev=25, azim=45)  # Ángulo de visualización

# Añadir grid y fondo
ax.grid(True, linestyle='--', alpha=0.5)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

plt.tight_layout()
plt.show()