import numpy as np
from scipy.spatial import cKDTree
import cupy as cp
from tqdm import tqdm

#_____________________________________________________________________________________________________
#               1] Funcion para distribuir particulas por el espacio de manera cilindrica

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
    z = np.random.uniform(0, L, N)

    # Conversion a coordenadas cartesianas
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Arreglo con coordenadas XYZ en formato Nx3
    s = np.vstack((x,y,z)).T.astype(np.float32)

    return s

#_____________________________________________________________________________________________________
#               2] Tratamiento del campo electrico calculado con LaPlace

E_Field_File = np.load("Electric_Field_np.npy") # Archivo numpy con E calculado

"""
    Interpolator_Electric_Field:

    Mediante E_Field_File que es el campo electrico generado con LaPlace, se crean 4 variables de
    interpolacion las cuales son Ex_values, Ey_values, Ez_values y tree. Estas ultima es creada 
    con cKDTree, un metodo desarrollado precisamente para encontrar el valor numerico de un campo 
    en un punto arbitrario de su espacio. Con esto nos sera sencillo asociar un valor de campo 
    electrico a una particula en el punto (X,Y,Z) cualquiera.

    Interpolate_E:

    Recibe la posicion [s] de una particula y en base a las 4 variables de interpolacion retorna
    un valor E correspondiente al campo electrico asociado a esa particula

    E_Field_File -> Campo electrico resultado de Laplace_E_Solution.py
"""

def Interpolator_Electric_Field(E_Field_File):

    # Extrayendo las coordenadas espaciales(X,Y,Z) del campo electrico
    points = E_Field_File[:, :3]

    # Extrayendo los valores del campo electrico en cada eje coordenado
    Ex_values = E_Field_File[:, 3]  # Campo Ex
    Ey_values = E_Field_File[:, 4]  # Campo Ey
    Ez_values = E_Field_File[:, 5]  # Campo Ez

    # Creando la funcion de interpolacion para cada campo electrico(Ex, Ey, Ez)
    tree = cKDTree(points)

    return tree, Ex_values, Ey_values, Ez_values

def Interpolate_E(tree, Ex_values, Ey_values, Ez_values, s):

    # Busca el indice(idx) del campo electrico correspondiente al punto [s]
    _, idx = tree.query(s)

    # Obtener los valores de Ex, Ey, Ez segun el indice [idx]
    Ex = Ex_values[idx]
    Ey = Ey_values[idx]
    Ez = Ez_values[idx]

    # retorna el campo electrico en un solo vector/matriz transpuesto
    return np.vstack((Ex, Ey, Ez)).T 

#_____________________________________________________________________________________________________
#               3] Parámetros de simulación

N = 1000000  # Número de partículas
dt = 0.03  # Delta de tiempo
q_m = 1.0  # Valor Carga/Masa

Rin = 20 # Radio interno del cilindro hueco
Rex = 40 # Primer radio externo del cilindro hueco
L = 60 # Longitud del cilindro

tree, Ex_values, Ey_values, Ez_values = Interpolator_Electric_Field(E_Field_File)  # Campo eléctrico y su interpolador
B0 = 5.0  # Magnitud del campo magnético radial

#_____________________________________________________________________________________________________
#               4] Inicialización de partículas (posición y velocidad)

s = initialize_particles(N, Rin=Rin, Rex=Rex, L=L)  # Posiciones iniciales

# Definicion de velocidades con limites en cada eje
Vx_min, Vx_max = -1.0, 1.0
Vy_min, Vy_max = -0.5, 0.5
Vz_min, Vz_max = 0.0, 100.0

v_x = Vx_min + (Vx_max - Vx_min) * np.random.rand(N).astype(np.float32)
v_y = Vy_min + (Vy_max - Vy_min) * np.random.rand(N).astype(np.float32)
v_z = Vz_min + (Vz_max - Vz_min) * np.random.rand(N).astype(np.float32)

v = np.vstack((v_x, v_y, v_z)).T  # Juntamos los valores en una matriz Nx3

#_____________________________________________________________________________________________________
#               5] Parametros de simulacion

timesteps = 500  # Número de frames de la animación

# Inicializar almacenamiento de datos
all_positions = np.zeros((timesteps, N, 3), dtype=np.float32)

#_____________________________________________________________________________________________________
#               6] Función para mover partículas

# Se obtiene el campo electrico(por ahora constante)
E = Interpolate_E(tree, Ex_values, Ey_values, Ez_values, s)

# Se pasan las variables (s,v,E) a la GPU
s_gpu = cp.array(s)
v_gpu = cp.array(v)
E_gpu = cp.array(E)

def move_particles(s, v, dt, q_m, E, B0):

    # Definira el rango en el campo magnetico tiene valor [50,60] en Z
    mask_B = (s[:, 2] >= 50) & (s[:,2] <= 60)

    # Calcular el radio en el plano XY
    r = cp.sqrt((s[:, 0])**2 + (s[:, 1])**2) + 1e-6

    # Inicializando vectores de campo magnetico
    Bx, By, Bz = cp.zeros_like(s[:, 0]), cp.zeros_like(s[:, 0]), cp.zeros_like(s[:, 0])
    
    # Campo magnético radial hacia el centro del cilindro
    Bx[mask_B] = -B0 * ((s[mask_B, 0]) / r[mask_B])
    By[mask_B] = -B0 * ((s[mask_B, 1]) / r[mask_B])

    # Crear la matriz del campo magnético para todas las partículas
    B = cp.column_stack((Bx, By, Bz))  

    # Cálculo optimizado de la Fuerza de Lorentz
    F_Lorentz = cp.cross(v, B)

    # Actualizacion de velocidad
    v += q_m * (E + F_Lorentz) * dt

    # Actualizacion de posicion
    s += v * dt

    # Mascara que define los limites de simulacion [0,0,0] ^ [120,120,180]
    mask_out = (s[:, 0] < -60) | (s[:, 0] > 60) | \
               (s[:, 1] < -60) | (s[:, 1] > 60) | \
               (s[:, 2] < 0) | (s[:, 2] > 180)
    
    # Cantidad de particulas que deben re ingresar al sistema
    num_reinsert = int(cp.sum(mask_out).item()) 
    
    if num_reinsert > 0:

        # Generamos nuevas posiciones en el cilindro en (X,Y)
        r_new = cp.sqrt(cp.random.uniform(Rin**2, Rex**2, num_reinsert))
        theta_new = cp.random.uniform(0, 2*cp.pi, num_reinsert)

        # Pasamos a coordenadas cartesianas
        x_new = r_new * cp.cos(theta_new)
        y_new = r_new * cp.sin(theta_new)
        z_new = cp.full(num_reinsert, 1, dtype=cp.float32)  # Z siempre en 180

        # Asignamos las nuevas posiciones
        s[mask_out, 0] = x_new
        s[mask_out, 1] = y_new
        s[mask_out, 2] = z_new

        # Asignamos nuevas velocidades aleatorias
        v_x_new = Vx_min + (Vx_max - Vx_min) * cp.random.rand(num_reinsert).astype(cp.float32)
        v_y_new = Vy_min + (Vy_max - Vy_min) * cp.random.rand(num_reinsert).astype(cp.float32)
        v_z_new = Vz_min + (Vz_max - Vz_min) * cp.random.rand(num_reinsert).astype(cp.float32)

        # Juntamos las velocidades en una matriz (num_reinsert, 3)
        v_new = cp.column_stack((v_x_new, v_y_new, v_z_new))

        # Asignamos las nuevas velocidades a las partículas reinsertadas
        v[mask_out] = v_new
    
    # Retornamos la posicion espacial de las particulas para ser almacenadas
    return s

#_____________________________________________________________________________________________________
#               7] Simulacion y proceso de renderizado

# Ejecutar la simulación y guardar datos con barra de progreso
print("Ejecutando simulación y guardando datos...")
for t in tqdm(range(timesteps), desc="Progreso"):

    # Funcion de movimiento
    s = move_particles(s_gpu, v_gpu, dt, q_m, E_gpu, B0)

    # Conversion de [s] a datos de CPU(numpy) nuevamente
    s_np = cp.asnumpy(s)

    # Guardar la posición de las partículas en este frame
    all_positions[t] = s_np  

# Guardar el archivo con todas las posiciones simuladas
np.save("particle_simulation.npy", all_positions)
print("✅ Simulación guardada exitosamente en 'particle_simulation.npy'")