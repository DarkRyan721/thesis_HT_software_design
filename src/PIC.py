import numpy as np
from scipy.spatial import cKDTree
import cupy as cp
from tqdm import tqdm
from thermostat import aplicar_termostato

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

E_Field_File = np.load("data_files/Electric_Field_np.npy") # Archivo numpy con E calculado
M_Field_File = np.load("data_files/Magnetic_Field_np.npy") # Archivo numpy con M calculado

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
    _, idx = tree.query(s.get())

    # Obtener los valores de Ex, Ey, Ez segun el indice [idx]
    Ex = Ex_values[idx]
    Ey = Ey_values[idx]
    Ez = Ez_values[idx]

    # retorna el campo electrico en un solo vector/matriz transpuesto
    return np.vstack((Ex, Ey, Ez)).T 

def Interpolator_Magnetic_Field(M_Field_File):

    # Extrayendo las coordenadas espaciales(X,Y,Z) del campo electrico
    points = M_Field_File[:, :3]

    # Extrayendo los valores del campo electrico en cada eje coordenado
    Ex_values = M_Field_File[:, 3]  # Campo Ex
    Ey_values = M_Field_File[:, 4]  # Campo Ey
    Ez_values = M_Field_File[:, 5]  # Campo Ez

    # Creando la funcion de interpolacion para cada campo electrico(Ex, Ey, Ez)
    tree = cKDTree(points)

    return tree, Ex_values, Ey_values, Ez_values

def Interpolate_M(tree, Mx_values, My_values, Mz_values, s):

    # Busca el indice(idx) del campo electrico correspondiente al punto [s]
    _, idx = tree.query(s.get())

    # Obtener los valores de Ex, Ey, Ez segun el indice [idx]
    Mx = Mx_values[idx]
    My = My_values[idx]
    Mz = Mz_values[idx]

    # retorna el campo electrico en un solo vector/matriz transpuesto
    return np.vstack((Mx, My, Mz)).T 

#_____________________________________________________________________________________________________
#               3] Parámetros de simulación

N = 100000 # Número de partículas
dt = 0.0000001  # Delta de tiempo
q_m = 7.35e5 # Valor Carga/Masa
m = 2.18e-25

#Lectura de parametro geometricos (Archivo txt)
def leer_datos_archivo(ruta_archivo):
    datos = {}
    with open(ruta_archivo, "r") as archivo:
        for linea in archivo:
            # Verificamos que la línea contenga ':'
            if ":" in linea:
                clave, valor = linea.split(":", maxsplit=1)
                # Limpiamos espacios
                clave = clave.strip()
                valor = valor.strip()
                # Almacenar en el diccionario (conversión a entero o float)
                datos[clave] = float(valor)
    return datos
ruta = "data_files/geometry_parameters.txt"
info = leer_datos_archivo(ruta)

Rin = info.get("radio_interno",0) # Radio interno del cilindro hueco
Rex = info.get("radio_externo",0) # Primer radio externo del cilindro hueco
L = info.get("profundidad",0) # Longitud del cilindro

tree, Ex_values, Ey_values, Ez_values = Interpolator_Electric_Field(E_Field_File)  # Campo eléctrico y su interpolador
tree_m, Mx_values, My_values, Mz_values = Interpolator_Magnetic_Field(M_Field_File)
B0 = 1000  # Magnitud del campo magnético radial

#_____________________________________________________________________________________________________
#               4] Inicialización de partículas (posición y velocidad)

s = initialize_particles(N, Rin=Rin, Rex=Rex, L=L)  # Posiciones iniciales

# Definicion de velocidades con limites en cada eje
Vx_min, Vx_max = -0, 0
Vy_min, Vy_max = -0, 0
Vz_min, Vz_max = 0.0, 200.0

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

# Se pasan las variables (s,v,E) a la GPU
s_gpu = cp.array(s)
v_gpu = cp.array(v)

Temp = []

def move_particles(s, v, dt, q_m, B0):

    # # Definira el rango en el campo magnetico tiene valor [0,0.02] en Z
    # mask_B = (s[:, 2] >= 0) & (s[:,2] <= 0.02)

    # # Calcular el radio en el plano XY
    # r = cp.sqrt((s[:, 0])**2 + (s[:, 1])**2)

    # # Inicializando vectores de campo magnetico
    # Bx, By, Bz = cp.zeros_like(s[:, 0]), cp.zeros_like(s[:, 0]), cp.zeros_like(s[:, 0])
    
    # # Campo magnético radial hacia el centro del cilindro
    # Bx[mask_B] = -B0 * ((s[mask_B, 0]) / r[mask_B])
    # By[mask_B] = -B0 * ((s[mask_B, 1]) / r[mask_B])

    # # Crear la matriz del campo magnético para todas las partículas
    # B = cp.column_stack((Bx, By, Bz))  

    # mask_B = (s[:, 2] >= 0) & (s[:, 2] <= (1.1*L))
    # B = Interpolate_M(tree_m, Mx_values, My_values, Mz_values, s[mask_B])

    # F_Lorentz_filtered = cp.cross(v[mask_B], B)

    # # Opcional: Asignar los resultados a un arreglo completo
    # F_Lorentz = cp.zeros_like(v)
    # F_Lorentz[mask_B] = F_Lorentz_filtered

    # # Cálculo optimizado de la Fuerza de Lorentz
    # mask_E = (s[:, 2] >= 0) & (s[:, 2] <= L)  # Máscara para partículas en el rango [0, L] en x
    # E_filtered = Interpolate_E(tree, Ex_values, Ey_values, Ez_values, s[mask_E])
    # E = cp.zeros_like(v)
    # E[mask_E] = E_filtered
    E = Interpolate_E(tree, Ex_values, Ey_values, Ez_values, s)
    E = cp.array(E)

    # Actualizacion de velocidad
    v += q_m * (E) * dt

    # Actualizacion de posicion
    s += v * dt

    # Mascara para vigilar colisiones en el cilindro
    r_collision = cp.sqrt(s[:, 0]**2 + s[:, 1]**2)
    mask_collision = ((r_collision >= (Rex)) | (r_collision <= (Rin))) & (s[:, 2] > 0) & (s[:, 2] <= L)
    num_collisions = int(cp.sum(mask_collision).item()) 

    if num_collisions > 0:
        # Velocidad antes de la colisión
        v_before = v[mask_collision]

        # Vector normal unitario (radial hacia afuera)
        normal_vector = cp.zeros_like(v_before)
        normal_vector[:, 0] = s[mask_collision, 0] / r_collision[mask_collision]
        normal_vector[:, 1] = s[mask_collision, 1] / r_collision[mask_collision]
        normal_vector[:, 2] = 0  # componente axial es cero para pared lateral cilindrica

        # Proyección de la velocidad en la dirección normal
        v_normal = cp.sum(v_before * normal_vector, axis=1, keepdims=True) * normal_vector
        v_tangencial = v_before - v_normal

        # Calcular velocidades despues de colision (con α dado por tu compañero)
        alpha = 0.9  # ejemplo, recibelo de tu compañero o definelo externamente
        v_after_collision = v_tangencial - alpha * v_normal

        # Calculo de energia cinetica antes y despues
        E_before = 0.5 * m * cp.sum(v_before**2, axis=1)
        E_after = 0.5 * m * cp.sum(v_after_collision**2, axis=1)
        delta_E = E_before - E_after  # Esto es lo que envias al termostato

        lambda_value, temp_aux = aplicar_termostato(delta_E, num_collisions, dt, 100, 300, 8.617e-23)
        v_final_mag = v_before * lambda_value.reshape(-1, 1)
        Temp.append(temp_aux)
        # (Aquí envías delta_E a tu compañero y recibes la nueva magnitud |v_final|)
        # Ejemplo simulado:
        v_final_mag = cp.sqrt(cp.sum(v_after_collision**2, axis=1))  # Temporal
        # Real: v_final_mag = funcion_termostato(delta_E)

        # Corriges dirección (mantienes dirección calculada pero ajustas magnitud recibida)
        v_direction = v_after_collision / cp.linalg.norm(v_after_collision, axis=1, keepdims=True)
        v_corrected = v_direction * v_final_mag[:, cp.newaxis]

        # Actualizas velocidad tras la colisión
        v[mask_collision] = v_corrected

        r_exceso = r_collision[mask_collision] - Rex

        # 2) Mover la partícula de vuelta a la frontera
        #    Restando ese exceso en la dirección normal.
        s[mask_collision, 0] -= r_exceso * normal_vector[:, 0]
        s[mask_collision, 1] -= r_exceso * normal_vector[:, 1]

    # Mascara que define los limites de simulacion [0,0,0] ^ [120,120,180]
    mask_out = (s[:, 0] < -3*Rex) | (s[:, 0] > 3*Rex) | \
               (s[:, 1] < -3*Rex) | (s[:, 1] > 3*Rex) | \
               (s[:, 2] < 0) | (s[:, 2] > 3*Rex)
    
    # Cantidad de particulas que deben re ingresar al sistema
    num_reinsert = int(cp.sum(mask_out).item()) 
    
    if num_reinsert > 0:

        # Generamos nuevas posiciones en el cilindro en (X,Y)
        r_new = cp.sqrt(cp.random.uniform(Rin**2, Rex**2, num_reinsert))
        theta_new = cp.random.uniform(0, 2*cp.pi, num_reinsert)

        # Pasamos a coordenadas cartesianas
        x_new = r_new * cp.cos(theta_new)
        y_new = r_new * cp.sin(theta_new)
        z_new = cp.full(num_reinsert, 0, dtype=cp.float32)  # Z siempre en 180

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
    s = move_particles(s_gpu, v_gpu, dt, q_m, B0)

    # Conversion de [s] a datos de CPU(numpy) nuevamente
    s_np = cp.asnumpy(s)

    # Guardar la posición de las partículas en este frame
    all_positions[t] = s_np  

# Guardar el archivo con todas las posiciones simuladas
np.save("data_files/particle_simulation.npy", all_positions)
print("Simulación guardada exitosamente en 'particle_simulation.npy'")

# import mplcursors

# N_graph = len(Temp)
# tiempo = np.arange(N_graph) * dt

# fig, ax = plt.subplots(figsize=(9, 5))
# line, = ax.plot(tiempo, Temp, marker='o', markersize=3, linewidth=1.5)

# ax.set_xlabel('Tiempo [s]')
# ax.set_ylabel('Temperatura [K]')
# ax.set_title('Temperatura vs Tiempo')
# ax.grid(True)

# # Activar cursor interactivo para mostrar valores
# cursor = mplcursors.cursor(line, hover=True)

# # Formato del texto emergente
# @cursor.connect("add")
# def on_hover(sel):
#     sel.annotation.set(text=f'Tiempo: {sel.target[0]:.2f}s\nTemperatura: {sel.target[1]:.4f} K')

# plt.tight_layout()
# plt.show()
