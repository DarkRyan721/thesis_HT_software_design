import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Cargar posiciones
all_positions = np.load("data_files/particle_simulation.npy", mmap_mode="r")
frame = 450
frame_data = all_positions[frame]

# Filtrar iones (etiqueta 1)
mask_ions = frame_data[:, 3] == 1
ions_points = frame_data[mask_ions, :3]

# Extraer coordenadas X y Z
x = ions_points[:, 0]
z = ions_points[:, 2]

# Crear el gráfico
fig, ax = plt.subplots(figsize=(8, 6))

# Histograma 2D
h = ax.hist2d(x, z, bins=200, cmap='hot')
plt.colorbar(h[3], ax=ax, label='Número de iones')

# Añadir el rectángulo gris: X[-0.05, 0.05], Z[0, 0.02]
rect = patches.Rectangle((-0.05, 0), 0.10, 0.02, linewidth=1, edgecolor='black', facecolor='gray', alpha=1)
ax.add_patch(rect)

# Etiquetas y formato
ax.set_xlabel('X [m]')
ax.set_ylabel('Z [m]')
ax.set_title('Densidad de Iones proyectada en el plano XZ (Frame 450)')
ax.grid(True)
plt.tight_layout()
plt.show()
