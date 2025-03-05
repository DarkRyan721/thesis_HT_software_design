import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from numba import cuda
import numba

#_____________________________________________________________________________________________________
#                           Parametros de simulacion

N = 1000 # Numero de particulas
dt = 0.01 # Delta de tiempo
q_m = 1.0 # Valor Carga/Masa

E = np.array([0.0, 0.0, 5.0], dtype=np.float32) # Campo electrico
B0 = 2.0 # Magnitud del campo magnetico radial

#_____________________________________________________________________________________________________
#                           Inicializacion de particulas(r, v)

s = np.random.rand(N,3).astype(np.float32) * 2 - 1 # Ubicacion espacial de las particulas
v = np.random.rand(N,3).astype(np.float32) - 0.5 # Velocidad iniciales aleatorias
# v = np.zeros((N, 3), dtype=np.float32)  # Para velocidad inicial cero

#_____________________________________________________________________________________________________
#                           Copiar los datos a una GPU

s_device = cuda.to_device(s)
v_device = cuda.to_device(v)

#_____________________________________________________________________________________________________
#                           Kernel CUDA para simulacion de particulas

@cuda.jit
def move_particles(s, v, dt, q_m, E, B0):
    i = cuda.grid(1)

    if i < s.shape[0]:
        # Calculo del radio en plano X-Y
        r = (s[i, 0]**2 + s[i,1]**2)**0.5 + 1e-6

        # Campo magnetico radial hacia el centro del cilindro
        Bx = -B0 * (s[i,0]/r)
        By = -B0 * (s[i,1]/r)
        Bz = 0.0

        B = cuda.local.array(3, dtype=numba.float32)
        B[0], B[1], B[2] = Bx, By, Bz

        # Fuerza de Lorentz
        Fx = v[i, 1] * B[2] - v[i, 2] * B[1] # Vy*Bz - Vz*By
        Fy = v[i, 2] * B[0] - v[i, 0] * B[2] # Vz*Bx - Vx*Bz
        Fz = v[i, 0] * B[1] - v[i, 1] * B[0] # Vx*By - Vy*Bx

        # Actualizacion de la velocidad
        v[i, 0] += q_m * (E[0] + Fx) * dt
        v[i, 1] += q_m * (E[1] + Fy) * dt
        v[i, 2] += q_m * (E[2] + Fz) * dt

        # Actualizacion de la posicion
        s[i, 0] += v[i, 0] * dt
        s[i, 1] += v[i, 1] * dt
        s[i, 2] += v[i, 2] * dt

# Configuracion CUDA
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

# Matplotlib 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Puntos iniciales

sc = ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=1, color = "blue")

# Limites del grafico

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 8)
ax.set_title("Evolución de Partículas")

# Función de actualización para animación
def update(frame):
    global s_device, v_device
    
    # Ejecutar PIC en GPU con campo magnético radial
    move_particles[blocks_per_grid, threads_per_block](s_device, v_device, dt, q_m, E, B0)
    
    # Copiar datos de vuelta a CPU
    x_host = s_device.copy_to_host()
    
    # Actualizar la posición en la animación
    sc._offsets3d = (x_host[:, 0], x_host[:, 1], x_host[:, 2])
    sc.set_offsets(x_host[:, :2])  # Actualiza X e Y
    sc.set_3d_properties(x_host[:, 2], zdir='z')  # Actualiza Z
    return sc,

# Crear animación (blit=False para evitar errores en 3D)
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)

plt.show()