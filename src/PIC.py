import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import NearestNDInterpolator

#_____________________________________________________________________________________________________
#            Funcion para distribuir particulas por el espacio de manera cilindrica

Z_shift = 120 # Variable que desplaza el cilindro de particulas en Z
XY_shift = 60 # Variable que desplaza el centro del cilindro en XY a [60,60]

"""
    initialize_particles:

    Funcion encargada de recibir la geometria del cilindro(Rin, Rex, L) y generar N particulas en el
    espacio permitido definido por esta misma geometria.

    N -> Numero de particulas
    Rin -> Radio interno
    Rex -> Radio externo
    L -> Longitud del cilindro
"""
def initialize_particles(N, Rin, Rex, L):
    
    # Generacion uniforme de valores cilindricos(r, theta, z)
    r = np.sqrt(np.random.uniform(Rin**2, Rex**2, N))
    theta = np.random.uniform(0, 2*np.pi, N)
    z = np.random.uniform(0+Z_shift, L+Z_shift, N)

    # Conversion a coordenadas cartesianas
    x = r * np.cos(theta) + XY_shift
    y = r * np.sin(theta) + XY_shift

    # Arreglo con coordenadas XYZ en formato Nx3
    s = np.vstack((x,y,z)).T.astype(np.float32)

    return s

#_____________________________________________________________________________________________________
#           Tratamiento del campo electrico calculado con LaPlace

E_Field_File = np.load("Electric_Field_np.npy") # Archivo numpy con E calculado

"""
    Interpolator_Electric_Field:

    Mediante E_Field_File que es el campo electrico generado con LaPlace, se crean 3 funciones de
    interpolacion las cuales son Ex_interp, Ey_interp, Ez_interp. Estas funciones son creadas con
    NearestNDInterpolator, un metodo desarrollado precisamente para encontrar el valor numerico de
    un campo en un punto arbitrario de su espacio. Con esto nos sera sencillo asociar un valor de
    campo electrico a una particula en el punto (X,Y,Z) cualquiera.

    E_Field_File -> Campo electrico resultado de Laplace_E_Solution.py
"""

def Interpolator_Electric_Field(E_Field_File):

    # Extrayendo las coordenadas espaciales(X,Y,Z) del campo
    points = E_Field_File[:, :3]

    # Extrayendo los valores del campo electrico en cada eje coordenado
    Ex_values = E_Field_File[:, 3]  # Campo Ex
    Ey_values = E_Field_File[:, 4]  # Campo Ey
    Ez_values = E_Field_File[:, 5]  # Campo Ez

    # Creando la funcion de interpolacion para cada campo electrico(Ex, Ey, Ez)
    Ex_interp = NearestNDInterpolator(points, Ex_values)
    Ey_interp = NearestNDInterpolator(points, Ey_values)
    Ez_interp = NearestNDInterpolator(points, Ez_values)

    return Ex_interp, Ey_interp, Ez_interp

#_____________________________________________________________________________________________________
#                           Parámetros de simulación

N = 1000  # Número de partículas
dt = 0.01  # Delta de tiempo
q_m = 1.0  # Valor Carga/Masa

Rin = 20 # Radio interno del cilindro hueco
Rex = 40 # Primer radio externo del cilindro hueco
L = 60 # Longitud del cilindro

E_interpolators = Interpolator_Electric_Field(E_Field_File)  # Campo eléctrico
B0 = 5.0  # Magnitud del campo magnético radial

#_____________________________________________________________________________________________________
#                           Inicialización de partículas (posición y velocidad)

s = initialize_particles(N, Rin=Rin, Rex=Rex, L=L)  # Posiciones iniciales

# Definicion de velocidades con limites en cada eje

Vx_min, Vx_max = -1.0, 1.0
Vy_min, Vy_max = -0.5, 0.5
Vz_min, Vz_max = -100.0, 0.0

v_x = Vx_min + (Vx_max - Vx_min) * np.random.rand(N).astype(np.float32)
v_y = Vy_min + (Vy_max - Vy_min) * np.random.rand(N).astype(np.float32)
v_z = Vz_min + (Vz_max - Vz_min) * np.random.rand(N).astype(np.float32)

v = np.vstack((v_x, v_y, v_z)).T  # Juntamos los valores en una matriz Nx3

# v = np.zeros((N, 3), dtype=np.float32)  # Para velocidad inicial cero

#_____________________________________________________________________________________________________
#                           Función para mover partículas

def move_particles(s, v, dt, q_m, E_interpolators, B0):

    # Definira el rango en el campo magnetico tiene valor [120,130] en Z
    mask_B = (s[:, 2] >= 120) & (s[:,2] <= 130)

    # Calcular el radio en el plano XY
    r = np.sqrt((s[:, 0] - XY_shift)**2 + (s[:, 1] - XY_shift)**2) + 1e-6

    # Inicializando vectores de campo magnetico
    Bx = np.zeros(N, dtype=np.float32)
    By = np.zeros(N, dtype=np.float32)
    Bz = np.zeros(N, dtype=np.float32)  # No hay componente en Z
    
    # Campo magnético radial hacia el centro del cilindro
    Bx[mask_B] = -B0 * ((s[mask_B, 0] - XY_shift) / r[mask_B])
    By[mask_B] = -B0 * ((s[mask_B, 1] - XY_shift) / r[mask_B])

    # Producto cruz v × B (Fuerza de Lorentz)
    Fx = v[:, 1] * Bz - v[:, 2] * By
    Fy = v[:, 2] * Bx - v[:, 0] * Bz
    Fz = v[:, 0] * By - v[:, 1] * Bx

    # Recuperando las funciones de interpolacion de E
    Ex_interp, Ey_interp, Ez_interp = E_interpolators

    # Obteniendo el campo electrico correspondiente a cada una de las particulas en Ex, Ey y Ez
    Ex = Ex_interp(s[:, 0], s[:, 1], s[:, 2])
    Ey = Ey_interp(s[:, 0], s[:, 1], s[:, 2])
    Ez = Ez_interp(s[:, 0], s[:, 1], s[:, 2])

    # Unificando los campos electricos en una sola matriz E de Nx3
    E = np.vstack((Ex, Ey, Ez)).T

    # Actualizacion de la velocidad
    v[:, 0] += q_m * (E[:, 0] + Fx) * dt  # Ex en cada partícula
    v[:, 1] += q_m * (E[:, 1] + Fy) * dt  # Ey en cada partícula
    v[:, 2] += q_m * (E[:, 2] + Fz) * dt  # Ez en cada partícula

    # Actualizar posición
    s[:, 0] += v[:, 0] * dt
    s[:, 1] += v[:, 1] * dt
    s[:, 2] += v[:, 2] * dt

    # Mascara que define los limites de simulacion [0,0,0] ^ [120,120,180]
    mask_out = (s[:, 0] < 0) | (s[:, 0] > 120) | \
               (s[:, 1] < 0) | (s[:, 1] > 120) | \
               (s[:, 2] < 0) | (s[:, 2] > 180)
    
    num_reinsert = np.sum(mask_out)  # Contar cuántas partículas se reingresan
    
    if num_reinsert > 0:

        # Generamos nuevas posiciones en el cilindro en (X,Y)
        r_new = np.sqrt(np.random.uniform(Rin**2, Rex**2, num_reinsert))
        theta_new = np.random.uniform(0, 2*np.pi, num_reinsert)

        # Pasamos a coordenadas cartesianas
        x_new = r_new * np.cos(theta_new) + XY_shift
        y_new = r_new * np.sin(theta_new) + XY_shift
        z_new = np.full(num_reinsert, 180)  # Z siempre en 180

        # Asignamos las nuevas posiciones
        s[mask_out, 0] = x_new
        s[mask_out, 1] = y_new
        s[mask_out, 2] = z_new

        # Asignamos nuevas velocidades aleatorias
        v_x_new = Vx_min + (Vx_max - Vx_min) * np.random.rand(num_reinsert).astype(np.float32)
        v_y_new = Vy_min + (Vy_max - Vy_min) * np.random.rand(num_reinsert).astype(np.float32)
        v_z_new = Vz_min + (Vz_max - Vz_min) * np.random.rand(num_reinsert).astype(np.float32)

        # Juntamos las velocidades en una matriz (num_reinsert, 3)
        v_new = np.vstack((v_x_new, v_y_new, v_z_new)).T

        # Asignamos las nuevas velocidades a las partículas reinsertadas
        v[mask_out] = v_new

#_____________________________________________________________________________________________________
#                           Configurar Matplotlib 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Crear los puntos iniciales
sc = ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=1, color="blue")

# Ajustar límites del gráfico
ax.set_xlim(0, 120)
ax.set_ylim(0, 120)
ax.set_zlim(0, 180)
ax.set_title("Evolución de Partículas (CPU)")

#_____________________________________________________________________________________________________
#                           Función de actualización para animación

def update(frame):
    global s, v
    
    # Ejecutar la simulación en CPU
    move_particles(s, v, dt, q_m, E_interpolators, B0)

    # Actualizar la posición en la animación
    sc._offsets3d = (s[:, 0], s[:, 1], s[:, 2])
    sc.set_offsets(s[:, :2])  # Actualiza X e Y
    sc.set_3d_properties(s[:, 2], zdir='z')  # Actualiza Z
    return sc,

# Crear animación (blit=False para evitar errores en 3D)
ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=False)


plt.show()