import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

#_____________________________________________________________________________________________________
#                           Parámetros de simulación

N = 1000  # Número de partículas
dt = 0.01  # Delta de tiempo
q_m = 1.0  # Valor Carga/Masa

E = np.array([0.0, 0.0, 5.0], dtype=np.float32)  # Campo eléctrico en Z
B0 = 2.0  # Magnitud del campo magnético radial

#_____________________________________________________________________________________________________
#                           Inicialización de partículas (posición y velocidad)

s = np.random.rand(N, 3).astype(np.float32) * 2 - 1  # Posiciones iniciales
v = np.random.rand(N, 3).astype(np.float32) - 0.5    # Velocidades iniciales aleatorias
# v = np.zeros((N, 3), dtype=np.float32)  # Para velocidad inicial cero

#_____________________________________________________________________________________________________
#                           Función para mover partículas (Versión en CPU)

def move_particles_cpu(s, v, dt, q_m, E, B0):
    # Calcular el radio en el plano X-Y
    r = np.sqrt(s[:, 0]**2 + s[:, 1]**2) + 1e-6  # Evitar división por cero
    
    # Campo magnético radial hacia el centro del cilindro
    Bx = -B0 * (s[:, 0] / r)
    By = -B0 * (s[:, 1] / r)
    Bz = np.zeros(N, dtype=np.float32)  # No hay componente en Z

    # Producto cruz v × B (Fuerza de Lorentz)
    Fx = v[:, 1] * Bz - v[:, 2] * By
    Fy = v[:, 2] * Bx - v[:, 0] * Bz
    Fz = v[:, 0] * By - v[:, 1] * Bx

    # Actualizar velocidad
    v[:, 0] += q_m * (E[0] + Fx) * dt
    v[:, 1] += q_m * (E[1] + Fy) * dt
    v[:, 2] += q_m * (E[2] + Fz) * dt

    # Actualizar posición
    s[:, 0] += v[:, 0] * dt
    s[:, 1] += v[:, 1] * dt
    s[:, 2] += v[:, 2] * dt

#_____________________________________________________________________________________________________
#                           Configurar Matplotlib 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Crear los puntos iniciales
sc = ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=1, color="blue")

# Ajustar límites del gráfico
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 8)
ax.set_title("Evolución de Partículas (CPU)")

#_____________________________________________________________________________________________________
#                           Función de actualización para animación

def update(frame):
    global s, v
    
    # Ejecutar la simulación en CPU
    move_particles_cpu(s, v, dt, q_m, E, B0)

    # Actualizar la posición en la animación
    sc._offsets3d = (s[:, 0], s[:, 1], s[:, 2])
    sc.set_offsets(s[:, :2])  # Actualiza X e Y
    sc.set_3d_properties(s[:, 2], zdir='z')  # Actualiza Z
    return sc,

# Crear animación (blit=False para evitar errores en 3D)
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)


plt.show()
