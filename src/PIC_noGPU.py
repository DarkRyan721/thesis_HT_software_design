import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import math
import pickle

# --------------------------------------------------------------------------------
# Lectura de geometría
def leer_datos_archivo(ruta_archivo):
    datos = {}
    with open(ruta_archivo, "r") as archivo:
        for linea in archivo:
            if ":" in linea:
                clave, valor = linea.split(":", maxsplit=1)
                clave = clave.strip()
                valor = valor.strip()
                datos[clave] = float(valor)
    return datos

info = leer_datos_archivo("data_files/geometry_parameters.txt")
Rin = info.get("radio_interno", 0)
Rex = info.get("radio_externo", 0)
L   = info.get("profundidad", 0)
h = info.get("lado_cubo", 0)

Nr = Rex
Nz = L
r_array = np.linspace(Rin, Rex, Nr)
z_array = np.linspace(0, L, Nz)

# n_e(r,z) => un array (Nr, Nz)
n_e = np.zeros((Nr, Nz), dtype=float)


def update_electron_density(n_e, r_array, z_array, dt, t):
    """
    Ejemplo simplificado de actualización de n_e sin resolver PDE real.
    Solo para ilustrar la estructura.
    """
    Nr, Nz = n_e.shape
    for ir in range(Nr):
        for iz in range(Nz):
            r = r_array[ir]
            z = z_array[iz]
            # Ejemplo: incrementamos algo en la región interior y
            # reducimos en la pared, de forma ficticia
            if r < (Rex * 0.7):
                n_e[ir, iz] += 5e14 * dt  # "crece" un poco
            else:
                n_e[ir, iz] *= 0.99      # "disminuye" un poco
    # Podrías imponer boundaries, etc.
    return n_e


# --------------------------------------------------------------------------------
# Inicializar partículas en un dominio cilíndrico
def initialize_particles(N, Rin, Rex, L):
    r = np.sqrt(np.random.uniform(Rin**2, Rex**2, N))
    theta = np.random.uniform(0, 2*np.pi, N)
    z = np.random.uniform(0, L, N)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    s = np.vstack((x,y,z)).T.astype(np.float32)
    return s

# --------------------------------------------------------------------------------
# Interpoladores de campo E y B fijos (ya calculados)
E_Field_File = np.load("data_files/Electric_Field_np.npy")
M_Field_File = np.load("data_files/Magnetic_Field_np.npy")

def Interpolator_Electric_Field(E_Field_File):
    points = E_Field_File[:, :3]
    Ex_values = E_Field_File[:, 3]
    Ey_values = E_Field_File[:, 4]
    Ez_values = E_Field_File[:, 5]
    tree = cKDTree(points)
    return tree, Ex_values, Ey_values, Ez_values

def Interpolate_E(tree, Ex_values, Ey_values, Ez_values, s):
    # s: Nx3
    _, idx = tree.query(s)  # query con NumPy
    Ex = Ex_values[idx]
    Ey = Ey_values[idx]
    Ez = Ez_values[idx]
    return np.vstack((Ex, Ey, Ez)).T

def Interpolator_Magnetic_Field(M_Field_File):
    points = M_Field_File[:, :3]
    Mx_values = M_Field_File[:, 3]
    My_values = M_Field_File[:, 4]
    Mz_values = M_Field_File[:, 5]
    tree = cKDTree(points)
    return tree, Mx_values, My_values, Mz_values

def Interpolate_M(treeM, Mx_values, My_values, Mz_values, s):
    _, idx = treeM.query(s)  # NumPy, no .get()
    Mx = Mx_values[idx]
    My = My_values[idx]
    Mz = Mz_values[idx]
    return np.vstack((Mx, My, Mz)).T

treeE, Ex_vals, Ey_vals, Ez_vals = Interpolator_Electric_Field(E_Field_File)
treeM, Mx_vals, My_vals, Mz_vals = Interpolator_Magnetic_Field(M_Field_File)

# --------------------------------------------------------------------------------
# Parámetros
N = 10000
dt = 0.00007
q_m = 1.0
m_ion = 2.18e-25

timesteps = 500

# Rango de velocidades iniciales
Vx_min, Vx_max =  0.0, 0.0
Vy_min, Vy_max =  0.0, 0.0
Vz_min, Vz_max = 75.0, 100.0

# --------------------------------------------------------------------------------
# Partículas iniciales (iones)
s = initialize_particles(N, Rin, Rex, L)
v_x = np.random.uniform(Vx_min, Vx_max, N).astype(np.float32)
v_y = np.random.uniform(Vy_min, Vy_max, N).astype(np.float32)
v_z = np.random.uniform(Vz_min, Vz_max, N).astype(np.float32)
v = np.column_stack((v_x, v_y, v_z))

# Etiquetas iniciales: -1 a todas para indicar que nacieron en "t=-1"
label = np.full(N, -1, dtype=np.int32)

# --------------------------------------------------------------------------------
# Definir las funciones de densidad y frecuencia de ionización (electrones + neutrales)
kB = 1.380649e-23
qe = 1.60217662e-19
me = 9.10938356e-31

ne0 = 1e17
nn0 = 1e19
Te0 = 10.0

def electron_density(x, y, z):
    return ne0

def neutral_density(x, y, z):
    return nn0

def electron_temperature(x, y, z):
    return Te0

def sigma_ionization(e_eV):
    return 1e-19  # valor ficticio

def ionization_frequency(x, y, z):
    Te_eV = electron_temperature(x,y,z)
    nn_local = neutral_density(x,y,z)
    Te_J = Te_eV * qe
    # velocidad electron ~ sqrt(8*kB*Te / (pi*m_e))  -- ojo con factor de conversión eV->J
    v_e = math.sqrt( (8.0 * kB * Te_J)/(math.pi * me) )
    sigma_ion = sigma_ionization(Te_eV)
    return nn_local * sigma_ion * v_e

# --------------------------------------------------------------------------------
# Ionización por muestreo de 1000 puntos
def ionize_particles(dt):
    # muestreamos Ntest puntos
    Ntest = 1000
    # Volumen cilíndrico
    volume = math.pi * (Rex**2 - Rin**2) * L

    r_rand = np.sqrt(np.random.uniform(Rin**2, Rex**2, Ntest))
    theta_rand = np.random.uniform(0, 2*np.pi, Ntest)
    z_rand = np.random.uniform(0, L, Ntest)

    x_test = r_rand * np.cos(theta_rand)
    y_test = r_rand * np.sin(theta_rand)
    z_test = z_rand

    new_positions = []
    new_velocities = []

    for i in range(Ntest):
        xx, yy, zz = x_test[i], y_test[i], z_test[i]
        nu_loc = ionization_frequency(xx, yy, zz)
        P_ion = 1.0 - np.exp(-nu_loc * dt)
        if np.random.rand() < P_ion:
            # se crea 1 ión
            new_positions.append([xx, yy, zz])
            new_velocities.append([0.0, 0.0, 0.0])

    if len(new_positions) == 0:
        return None, None

    new_positions = np.array(new_positions, dtype=np.float32)
    new_velocities = np.array(new_velocities, dtype=np.float32)
    return new_positions, new_velocities

# --------------------------------------------------------------------------------
# Función para mover partículas (E, B fijos) + colisiones con paredes + reinsertar
def move_particles(s, v, dt, q_m):
    """
    Mover partículas con E y B fijos.
    NO se hace la ionización aquí, sino en la función 'ionize_particles'.
    Si quieres meter colisión ión--neutral aquí, adelante.
    """
    # Interpolar E y B
    E = Interpolate_E(treeE, Ex_vals, Ey_vals, Ez_vals, s)
    B = Interpolate_M(treeM, Mx_vals, My_vals, Mz_vals, s)

    # Integrador BORIS (recomendable con B),
    # pero para simplificar, lo haremos con el naive E:
    # v <- v + (q/m) * E * dt
    v += q_m * E * dt

    # O si quisieras un integrador decente, implementa BORIS aquí

    # Actualizar posición
    s += v * dt

    # Colisión con paredes
    r = np.sqrt(s[:,0]**2 + s[:,1]**2)
    mask_wall = ((r>=Rex) | (r<=Rin)) & (s[:,2]>=0) & (s[:,2]<=L)
    if np.any(mask_wall):
        handle_wall_collisions(s, v, mask_wall)

    # Reinsertar
    mask_out = (s[:,0]<-3*Rex)|(s[:,0]>3*Rex)|(s[:,1]<-3*Rex)|(s[:,1]>3*Rex)|(s[:,2]<0)|(s[:,2]>3*Rex)
    if np.any(mask_out):
        reinsert_particles(s, v, mask_out)

    return s, v

def handle_wall_collisions(s, v, mask_wall):
    # Rebote inelástico con coef. alpha
    alpha = 0.9
    r = np.sqrt(s[:,0]**2 + s[:,1]**2)
    normal = np.zeros_like(s)
    normal[:,0] = s[:,0]/(r+1e-14)
    normal[:,1] = s[:,1]/(r+1e-14)

    vw = v[mask_wall]
    nw = normal[mask_wall]

    v_normal = np.sum(vw*nw, axis=1, keepdims=True)*nw
    v_tang = vw - v_normal
    v_after = v_tang - alpha*v_normal

    v[mask_wall] = v_after

    # Empujar partícula a la frontera
    s[mask_wall,0] -= (r[mask_wall]-Rex)*nw[:,0]
    s[mask_wall,1] -= (r[mask_wall]-Rex)*nw[:,1]

def reinsert_particles(s, v, mask_out):
    num_out = np.count_nonzero(mask_out)
    r_new = np.sqrt(np.random.uniform(Rin**2, Rex**2, num_out))
    th_new = np.random.uniform(0, 2*np.pi, num_out)
    x_new = r_new*np.cos(th_new)
    y_new = r_new*np.sin(th_new)
    z_new = np.zeros(num_out, dtype=np.float32)

    s[mask_out,0] = x_new
    s[mask_out,1] = y_new
    s[mask_out,2] = z_new

    # Velocidades aleatorias
    v_x_new = np.random.uniform(Vx_min, Vx_max, num_out).astype(np.float32)
    v_y_new = np.random.uniform(Vy_min, Vy_max, num_out).astype(np.float32)
    v_z_new = np.random.uniform(Vz_min, Vz_max, num_out).astype(np.float32)
    v_new = np.column_stack((v_x_new,v_y_new,v_z_new))
    v[mask_out] = v_new

# --------------------------------------------------------------------------------
# Bucle principal
positions_list = []  # para guardar la posición de cada timestep
velocities_list = []
labels_list = []

for t in tqdm(range(timesteps), desc="Simulando"):

    # (1) Mover iones
    s, v = move_particles(s, v, dt, q_m)

    # (2) Generar iones nuevos por colisión electron--neutral
    new_pos, new_vel = ionize_particles(dt)

    if new_pos is not None:
        # Cantidad de partículas nuevas
        newN = new_pos.shape[0]

        # 2.1) Apilar posiciones y velocidades
        s = np.vstack((s, new_pos))
        v = np.vstack((v, new_vel))

        # 2.2) Crear etiquetas para las nuevas partículas
        new_label = np.full(newN, t, dtype=np.int32)  # nacen en timestep t
        label = np.hstack((label, new_label))

    # (3) Guardar en listas (o arrays)
    positions_list.append(s.copy())
    velocities_list.append(v.copy())
    labels_list.append(label.copy())

with open("data_files/particle_simulation.pkl", "wb") as f:
    pickle.dump({
        "positions": positions_list,
        "velocities": velocities_list,
        "labels": labels_list
    }, f)
# Ejemplo de guardado final
# Ojo, s y v cambian de tamaño en cada timestep
# Podrías picklear la lista, o guardarlo como .npy en un for:
# np.save("data_files/particle_sim_step%d.npy"%t, s)
