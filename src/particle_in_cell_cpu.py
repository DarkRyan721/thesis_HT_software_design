import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from monte_carlo_collisions import MCC_Numpy
from project_paths import data_file

class PIC():
    def __init__(self, Rin, Rex, N, L, dt, q_m, alpha, sigma_ion):
        self.Rin = Rin
        self.Rex = Rex
        self.N = N
        self.L = L
        self.dt = dt
        self.q_m = q_m
        self.alpha = alpha
        self.sigma_ion = sigma_ion

        self.E_values = np.load(data_file("E_Field_Poisson.npy"))[:,3:]
        self.mesh_nodes = np.load(data_file("E_Field_Poisson.npy"))[:,:3]
        self.B_values = np.load(data_file("Magnetic_Field_np.npy"))[:,3:]
        self.Rho = np.load(data_file('density_n0.npy'))
        self.Rho_end = self.Rho

        self.tree = cKDTree(self.mesh_nodes)
        self.w_ion = 7.75e13 / self.N

    def initialize_neutros(self):
        r = np.sqrt(np.random.uniform(self.Rin**2, self.Rex**2, self.N))
        theta = np.random.uniform(0, 2*np.pi, self.N)
        z = np.random.uniform(0, self.L, self.N)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        self.s = np.vstack((x,y,z)).T.astype(np.float32)

    def interpolate_to_S(self, s_particle):
        _, idx = self.tree.query(s_particle)
        # print(s_particle.shape, self.E_values.shape, self.B_values.shape)
        return self.E_values[idx], self.B_values[idx]

    def initizalize_to_simulation(self, v_neutro, timesteps):
        self.initialize_neutros()
        self.s_label = np.zeros((self.N,1), dtype=np.float32)
        self.Vz_min = v_neutro - (0.05*v_neutro)
        self.Vz_max = v_neutro + (0.05*v_neutro)
        v_z = self.Vz_min + (self.Vz_max - self.Vz_min) * np.random.rand(self.N).astype(np.float32)
        v_x = np.zeros_like(v_z)
        v_y = np.zeros_like(v_z)
        self.v = np.vstack((v_x, v_y, v_z)).T
        self.timesteps = timesteps
        self.all_positions = np.zeros((timesteps, self.N, 4), dtype=np.float32)

    def rho_update(self):
        s_label_cpu = self.s_label.ravel()
        _, idxs = self.tree.query(self.s)
        pesos = np.where(s_label_cpu == 1, self.w_ion, 0.0)
        nuevo_rho = np.bincount(idxs, weights=pesos, minlength=len(self.mesh_nodes))
        self.Rho_end = self.Rho + nuevo_rho

    def move_particles(self):
        self.s_label = MCC_Numpy(s=self.s, v=self.v, s_label=self.s_label, rho=self.Rho_end,
                           sigma_ion=self.sigma_ion, dt=self.dt, tree=self.tree)
        self.rho_update()

        mask_ion = (self.s_label.ravel() == 1)
        s_ion = self.s[mask_ion]

        E, B = self.interpolate_to_S(s_ion)
        F_Lorentz = np.cross(self.v[mask_ion], B)
        self.v[mask_ion] += self.q_m * (E + F_Lorentz) * self.dt
        self.s += self.v * self.dt

        r_collision = np.sqrt(self.s[:, 0]**2 + self.s[:, 1]**2)
        mask_collision = ((r_collision >= self.Rex) | (r_collision <= self.Rin)) & \
                         (self.s[:, 2] > 0) & (self.s[:, 2] <= self.L)

        if np.any(mask_collision):
            v_before = self.v[mask_collision]
            normal_vector = np.zeros_like(v_before)
            normal_vector[:, 0] = self.s[mask_collision, 0] / r_collision[mask_collision]
            normal_vector[:, 1] = self.s[mask_collision, 1] / r_collision[mask_collision]
            v_normal = np.sum(v_before * normal_vector, axis=1, keepdims=True) * normal_vector
            v_tangencial = v_before - v_normal
            v_after_collision = v_tangencial - self.alpha * v_normal
            v_magnitude = np.sqrt(np.sum(v_after_collision**2, axis=1))
            v_direction = v_after_collision / np.linalg.norm(v_after_collision, axis=1, keepdims=True)
            v_corrected = v_direction * v_magnitude[:, np.newaxis]
            self.v[mask_collision] = v_corrected

            r_exceso = r_collision[mask_collision] - self.Rex
            self.s[mask_collision, 0] -= r_exceso * normal_vector[:, 0]
            self.s[mask_collision, 1] -= r_exceso * normal_vector[:, 1]

        mask_out = (self.s[:, 0] < -0.187) | (self.s[:, 0] > 0.187) | \
                   (self.s[:, 1] < -0.187) | (self.s[:, 1] > 0.187) | \
                   (self.s[:, 2] < 0) | (self.s[:, 2] > 0.2)

        num_reinsert = int(np.sum(mask_out))

        if num_reinsert > 0:
            r_new = np.sqrt(np.random.uniform(self.Rin**2, self.Rex**2, num_reinsert))
            theta_new = np.random.uniform(0, 2*np.pi, num_reinsert)
            x_new = r_new * np.cos(theta_new)
            y_new = r_new * np.sin(theta_new)
            z_new = np.full(num_reinsert, 0, dtype=np.float32)

            self.s[mask_out, 0] = x_new
            self.s[mask_out, 1] = y_new
            self.s[mask_out, 2] = z_new

            v_z_new = self.Vz_min + (self.Vz_max - self.Vz_min) * np.random.rand(num_reinsert).astype(np.float32)
            v_x_new = np.zeros_like(v_z_new)
            v_y_new = np.zeros_like(v_z_new)

            v_new = np.column_stack((v_x_new, v_y_new, v_z_new))
            self.v[mask_out] = v_new
            self.s_label[mask_out] = 0

        mask_ISP = (self.s_label.ravel() == 1) & (self.v[:, 2] > 0) & (self.s[:, 2] > self.L)

        if int(np.sum(mask_ISP).item()) > 0:
            v_ion = self.v[mask_ISP][:, 2]
            self.specific_impulse = np.mean(v_ion) / 9.806
        else:
            self.specific_impulse = 0.0

    def render(self):
        """
        render:

        Funcion encargada de renderizar el movimiento de las particulas y posteriormente guardar este mismo
        """

        #___________________________________________________________________________________________
        #       Ciclo de trabajo de renderizado

        print("Ejecutando simulación y guardando datos...")
        for t in tqdm(range(self.timesteps), desc="Progreso"):
            # Funcion de movimiento
            self.move_particles()

            combined = np.concatenate((self.s, self.s_label), axis=1)

            # Guardar la posición de las partículas en este frame
            self.all_positions[t] = combined

        #___________________________________________________________________________________________
        #       Guardar el archivo con todas las posiciones simuladas

        np.save(data_file("particle_simulation.npy"), self.all_positions)
        print("Simulación guardada exitosamente en 'particle_simulation.npy'")

        print("Impulso Especifico: ", self.specific_impulse*0.2)

        np.save(data_file("density_end.npy"), self.Rho_end)
        print("Densidad guardada exitosamente en  'density_end.npy'")

        #___________________________________________________________________________________________

if __name__ == "__main__":
    N = 1000
    dt = 0.00000004
    q_m = 7.35e5
    alpha = 0.9
    frames = 500
    sigma_ion = 1e-11

    # def leer_datos_archivo(ruta_archivo):
    #     datos = {}
    #     with open(ruta_archivo, "r") as archivo:
    #         for linea in archivo:
    #             # Verificamos que la línea contenga ':'
    #             if ":" in linea:
    #                 clave, valor = linea.split(":", maxsplit=1)
    #                 # Limpiamos espacios
    #                 clave = clave.strip()
    #                 valor = valor.strip()
    #                 # Almacenar en el diccionario (conversión a entero o float)
    #                 datos[clave] = float(valor)
    #     return datos
    # ruta = data_file("geometry_parameters.txt")
    # info = leer_datos_archivo(ruta)

    # Rin = info.get("radio_interno",0) # Radio interno del cilindro hueco
    # Rex = info.get("radio_externo",0) # Primer radio externo del cilindro hueco
    # L = info.get("profundidad",0) # Longitud del cilindro

    # EXAMPLE FOR GUI:

    """
        * Entradas que brinda el usuario para el renderizado:

        N -> Numero de particulas
        Rin -> Radio interno
        Rex -> Radio externo
        L -> Longitud/profundidad del cilindro
        frames -> cantidad de frames que desea

        * Valores que cambian segun el caso(NO SE LES PIDE AL USUARIO):

        dt -> Es el delta de tiempo con el que transcurre la simulacion
        q_m -> Valor de carga-masa del neutro
        alpha -> valor de reduccion de velocidad por colision
        sigma_ion -> Concurrencia de las ionizacion/neutralizaciones
        V_neutro -> velocidad inicial de los neutros
    """

    # 1. Crear el objeto de PIC con todas sus variables

    pic = PIC(Rin=0.028, Rex=0.05, N=N, L=0.02, dt=dt, q_m=q_m, alpha=alpha, sigma_ion=sigma_ion)

    # 2. Inicializar la simulacion(Posiciones iniciales, velocidades iniciales...)

    pic.initizalize_to_simulation(v_neutro=200, timesteps=frames)

    # 3. Realizar el render

    pic.render()