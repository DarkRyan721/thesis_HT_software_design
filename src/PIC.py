import numpy as np
from scipy.spatial import cKDTree
import cupy as cp
from tqdm import tqdm
from thermostat import aplicar_termostato
from scipy.spatial import cKDTree
from MCC import MCC

class PIC():
    def __init__(self, Rin, Rex, N, L, dt, q_m, alpha, sigma_ion):
        """
        Aqui se inicializaran todos los parametros geometricos e iniciales para el correcto funcionamiento de PIC.

        Rin -> Radio interno
        Rex -> Radio externo
        N -> Numero de particulas
        L -> Longitud del cilindro
        dt -> Es el delta de tiempo con el que transcurre la simulacion
        q_m -> Valor de carga-masa del neutro
        alpha -> valor de reduccion de velocidad por colision
        E_values -> los valores espaciales del campo electrico [Mx3] -> [Ex, Ey, Ez]
        B_values -> los valores espaciales del campo magnetico [Mx3] -> [Bx, By, Bz]
        mesh_nodes -> la posicion espacial de cada nodo de la malla [Mx3] -> [X, Y, Z]
        tree -> Es un objeto interpolador creado con la libreria scipy

        """

        #___________________________________________________________________________________________
        #       Inicializacion de variables con valores de entrada

        self.Rin = Rin #[m]
        self.Rex = Rex #[m]
        self.N = N #[1]
        self.L = L #[m]
        self.dt = dt #[s]
        self.q_m = q_m #[C/kg]
        self.alpha = alpha #[1]
        self.sigma_ion = sigma_ion #[1]

        #___________________________________________________________________________________________
        #       Cargando y guardando los valores de Campo Electrico

        E_Field_File = np.load("data_files/Electric_Field_np.npy") # Archivo numpy con E calculado

        self.E_values = E_Field_File[:,3:] #[Mx3]
        self.mesh_nodes = E_Field_File[:, :3] #[Mx3]

        #___________________________________________________________________________________________
        #       Cargando y guardando los valores de Campo Magnetico

        B_Field_File = np.load("data_files/Magnetic_Field_np.npy") # Archivo numpy con M calculado

        self.B_values = B_Field_File[:,3:] #[Mx3]

        #___________________________________________________________________________________________
        #       Cargando y guardando la densidad de electrones inicial

        self.Rho = np.load('data_files/density_n0.npy')

        self.Rho_end = self.Rho # Rho_end -> es la densidad de electrones en el tiempo.

        #___________________________________________________________________________________________
        #       Creacion de una herramienta de interpolacion para los puntos de los nodos

        self.tree = cKDTree(self.mesh_nodes)

        #___________________________________________________________________________________________
        #       Creacion de los valores de macroparticulas del ion

        self.w_ion = (7.75e13/self.N)

        #___________________________________________________________________________________________

    def initialize_neutros(self):
        """
        initialize_neutros:

        Funcion encargada de usar la geometria del propulsor(Rin, Rex, L) y generar N particulas en el
        espacio permitido definido por esta misma geometria. Creara la matriz espacial [s] la cual
        contiene la posicion espacial de las N particulas. la dimension de [S] es de Nx3
        """

        #___________________________________________________________________________________________
        #       Generacion uniforme de valores cilindricos(r, theta, z)

        r = np.sqrt(np.random.uniform(self.Rin**2, self.Rex**2, self.N))
        theta = np.random.uniform(0, 2*np.pi, self.N)
        z = np.random.uniform(0, self.L, self.N)

        #___________________________________________________________________________________________
        #       Conversion a coordenadas cartesianas
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        #___________________________________________________________________________________________
        # Arreglo con coordenadas XYZ en formato Nx3

        self.s = np.vstack((x,y,z)).T.astype(np.float32) #[Nx3]

        #___________________________________________________________________________________________

    def interpolate_to_S(self, s_particle):
        """
        interpolate_to_S:

        Funcion encargada de interpolar las posiciones de las particulas [S] a su nodo mas
        proximo en el espacio y asi hallar su valor de E y B.

        s_particle -> arreglo con la posicion espacial de las particulas
        """

        #___________________________________________________________________________________________
        #       Busca el punto (idx) mas cercano a la particula (s_particle)

        _, idx = self.tree.query(s_particle.get())

        #___________________________________________________________________________________________
        #       retorna el valor de campo electrico y magnetico en el punto [idx]

        return self.E_values[idx], self.B_values[idx]
    
        #___________________________________________________________________________________________
    
    def initizalize_to_simulation(self, v_neutro, timesteps):
        """
        initizalize_to_simulation:

        Funcion encargada de inicializar las particulas, sus etiquetas(iones o neutros) y velocidades
        ademas de las variables de almacenamiento del render[all_positions] o la cantidad
        de frames de la simulacion[timesteps]

        v_neutron -> Velocidad inicial de los neutros
        timesteps -> Cantidad de frames en la simulacion
        """

        #___________________________________________________________________________________________
        #       Se inicializan las posiciones de los neutrones junto con sus etiquetas

        self.initialize_neutros()

        self.s_label = cp.zeros((self.N,1))

        #___________________________________________________________________________________________
        #       Se inicializan las velocidad de los neutrones
        
        self.Vz_min = v_neutro - (0.05*v_neutro)
        self.Vz_max = v_neutro + (0.05*v_neutro)

        v_z = self.Vz_min + (self.Vz_max - self.Vz_min) * np.random.rand(self.N).astype(np.float32) #[m/s]
        v_x = np.zeros_like(v_z) #[m/s]
        v_y = np.zeros_like(v_z) #[m/s]

        self.v = np.vstack((v_x, v_y, v_z)).T #[m/s] 

        #___________________________________________________________________________________________
        #       Se establecen la cantidad de frames y se crea la variable del renderizado

        self.timesteps = timesteps #[1]
        self.all_positions = np.zeros((timesteps, self.N, 4), dtype=np.float32)

        #___________________________________________________________________________________________

    def rho_update(self):
        """
        rho_update:

        Actualiza la densidad de electrones (rho) basada en la distribución actual
        de partículas ionizadas en la malla.
        """

        #___________________________________________________________________________________________
        #       Convierte la posicion y etiquetas de las particulas a numpy

        s_cpu = cp.asnumpy(self.s)
        s_label_cpu = cp.asnumpy(self.s_label.ravel())

        #___________________________________________________________________________________________
        #       Encuentra el nodo mas proximo para cada particula

        _, idxs = self.tree.query(s_cpu)

        #___________________________________________________________________________________________
        #       Se identificaran cada macroparticula con su peso respectivo

        pesos = np.where(s_label_cpu == 1, self.w_ion, 0.0)

        #___________________________________________________________________________________________
        #       Se hace un conteo de macroparticulas a cada nodo mas proximo

        nuevo_rho = np.bincount(idxs, weights=pesos, minlength=len(self.mesh_nodes))

        #___________________________________________________________________________________________
        #       Se actualiza la densidad de electrones

        self.Rho_end = self.Rho + nuevo_rho

        #print("Nueva rho promedio:", np.mean(self.Rho_end)) #BORRAR

        #___________________________________________________________________________________________

    def move_particles(self):
        """
        move_particles:

        Funcion encargada de la logica de simulacion en cada frame. Encargada de llamar a MCC, aplicar
        las ecuaciones de PIC, verificar colisiones, re-inserciones de particulas y calculo del
        impulso especifico
        """

        #___________________________________________________________________________________________
        #       Se ionizan y neutralizan particulas

        self.s_label = MCC(s=self.s, v=self.v, s_label=self.s_label, rho=self.Rho_end, sigma_ion=self.sigma_ion, dt=self.dt, tree=self.tree)

        self.rho_update()

        #   print("Iones: ", cp.sum(self.s_label == 1)) #BORRAR

        #___________________________________________________________________________________________
        #       Se filtran los iones. Unicos afectados por campo electrico y magnetico

        mask_ion = (self.s_label.ravel() == 1)
        s_ion = self.s[mask_ion]

        #___________________________________________________________________________________________
        #       Se obtienen los valores de campo electrico y magnetico para los iones

        E, B = self.interpolate_to_S(s_ion)
        E = cp.asarray(E)

        F_Lorentz = cp.cross(self.v[mask_ion], B)

        #___________________________________________________________________________________________
        #       Se aplican las ecuaciones de PIC para la dinamica de las particulas

        self.v[mask_ion] += self.q_m * (E+F_Lorentz) * self.dt

        self.s += self.v * self.dt

        # mask_negative_z = (self.v[:, 2] < 0) & (self.s[:,2] >= 0.02) # Crea una máscara booleana donde Z es negativo
        # # Filtrar las partículas con valores negativos en Z
        # num_particles_negative_z = cp.sum(mask_negative_z).item()  # Cuenta cuántas partículas tienen valores negativos en Z
        # print(f"Cantidad de partículas con valores negativos en Z: {num_particles_negative_z}")

        #___________________________________________________________________________________________
        #       Se aplican las colisiones elasticas con la estructura del propulsor

        r_collision = cp.sqrt(self.s[:, 0]**2 + self.s[:, 1]**2)

        # Mascara para vigilar colisiones con el cilindro
        mask_collision = ((r_collision >= (self.Rex)) | (r_collision <= (self.Rin))) & (self.s[:, 2] > 0) & (self.s[:, 2] <= self.L)

        num_collisions = int(cp.sum(mask_collision).item()) 

        if num_collisions > 0:
            #___________________________________________________________________________________________
            #       Velocidad antes de la colisión

            v_before = self.v[mask_collision]

            #___________________________________________________________________________________________
            #       Vector normal unitario (radial hacia afuera)

            normal_vector = cp.zeros_like(v_before)
            normal_vector[:, 0] = self.s[mask_collision, 0] / r_collision[mask_collision]
            normal_vector[:, 1] = self.s[mask_collision, 1] / r_collision[mask_collision]
            normal_vector[:, 2] = 0

            #___________________________________________________________________________________________
            #       Proyección de la velocidad en la dirección normal

            v_normal = cp.sum(v_before * normal_vector, axis=1, keepdims=True) * normal_vector
            v_tangencial = v_before - v_normal

            #___________________________________________________________________________________________
            #       Calcular velocidades despues de colision (con α asignado)
            v_after_collision = v_tangencial - self.alpha * v_normal

            #___________________________________________________________________________________________
            #       Correccion de las direcciones

            v_magnitude = cp.sqrt(cp.sum(v_after_collision**2, axis=1))
            v_direction = v_after_collision / cp.linalg.norm(v_after_collision, axis=1, keepdims=True)
            v_corrected = v_direction * v_magnitude[:, cp.newaxis]

            #___________________________________________________________________________________________
            #       Actualizacion de la velocidad tras la colisión

            self.v[mask_collision] = v_corrected

            #___________________________________________________________________________________________
            #       Mover la partícula de vuelta a la frontera

            r_exceso = r_collision[mask_collision] - self.Rex

            self.s[mask_collision, 0] -= r_exceso * normal_vector[:, 0]
            self.s[mask_collision, 1] -= r_exceso * normal_vector[:, 1]

        #___________________________________________________________________________________________
        #       Mascara que define los limites de simulacion

        mask_out = (self.s[:, 0] < -0.187) | (self.s[:, 0] > 0.187) | \
                   (self.s[:, 1] < -0.187) | (self.s[:, 1] > 0.187) | \
                   (self.s[:, 2] < 0) | (self.s[:, 2] > 0.2)
        
        num_reinsert = int(cp.sum(mask_out).item())

        #print("Salieron: ", num_reinsert) #BORRAR

        if num_reinsert > 0:
            #___________________________________________________________________________________________
            #       Generamos nuevas posiciones en el cilindro en (X,Y)

            r_new = cp.sqrt(cp.random.uniform(self.Rin**2, self.Rex**2, num_reinsert))
            theta_new = cp.random.uniform(0, 2*cp.pi, num_reinsert)

            #___________________________________________________________________________________________
            #       Pasamos a coordenadas cartesianas

            x_new = r_new * cp.cos(theta_new)
            y_new = r_new * cp.sin(theta_new)
            z_new = cp.full(num_reinsert, 0, dtype=cp.float32)

            #___________________________________________________________________________________________
            #       Asignamos las nuevas posiciones

            self.s[mask_out, 0] = x_new
            self.s[mask_out, 1] = y_new
            self.s[mask_out, 2] = z_new

            #___________________________________________________________________________________________
            #       Asignamos nuevas velocidades aleatorias

            v_z_new = self.Vz_min + (self.Vz_max - self.Vz_min) * cp.random.rand(num_reinsert).astype(cp.float32)
            v_x_new = cp.zeros_like(v_z_new)
            v_y_new = cp.zeros_like(v_z_new)

            #___________________________________________________________________________________________
            #       Juntamos las velocidades en una matriz (num_reinsert, 3)

            v_new = cp.column_stack((v_x_new, v_y_new, v_z_new))

            #___________________________________________________________________________________________
            #       Asignamos las nuevas velocidades a las partículas reinsertadas

            self.v[mask_out] = v_new
            self.s_label[mask_out] = 0

        #___________________________________________________________________________________________
        #       Calculo del impulso especifico

        mask_ISP = (self.s_label.ravel() == 1) & (self.v[:, 2] > 0) & (self.s[:, 2] > self.L)

        if int(cp.sum(mask_ISP).item()) > 0:
            v_ion = self.v[mask_ISP][:, 2]
            self.specific_impulse = cp.mean(v_ion) / 9.806
        else:
            self.specific_impulse = 0.0

        #___________________________________________________________________________________________

    def render(self):
        """
        render:

        Funcion encargada de renderizar el movimiento de las particulas y posteriormente guardar este mismo
        """

        #___________________________________________________________________________________________
        #       Conversion de las posiciones y velocidades a cupy

        self.s = cp.array(self.s)
        self.v = cp.array(self.v)

        #___________________________________________________________________________________________
        #       Ciclo de trabajo de renderizado

        print("Ejecutando simulación y guardando datos...")
        for t in tqdm(range(self.timesteps), desc="Progreso"):
            # Funcion de movimiento
            self.move_particles()

            # Conversion de [s] a datos de CPU(numpy)
            s_np = cp.asnumpy(self.s)
            s_label_np = self.s_label.get()
            combined = np.concatenate((s_np, s_label_np), axis=1)

            # Guardar la posición de las partículas en este frame
            self.all_positions[t] = combined

        #___________________________________________________________________________________________
        #       Guardar el archivo con todas las posiciones simuladas

        np.save("data_files/particle_simulation.npy", self.all_positions)
        print("Simulación guardada exitosamente en 'particle_simulation.npy'")

        print("Impulso Especifico: ", self.specific_impulse)

        np.save("data_files/density_end.npy", self.Rho_end)
        print("Densidad guardada exitosamente en  'density_end.npy'")

        #___________________________________________________________________________________________

if __name__ == "__main__":
    N = 100000
    dt = 0.00000004
    q_m = 7.35e5
    alpha = 0.9
    frames = 500
    sigma_ion = 1e-11

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

    pic = PIC(Rin=Rin, Rex=Rex, N=N, L=L, dt=dt, q_m=q_m, alpha=alpha, sigma_ion=sigma_ion)

    # 2. Inicializar la simulacion(Posiciones iniciales, velocidades iniciales...)

    pic.initizalize_to_simulation(v_neutro=200, timesteps=frames)

    # 3. Realizar el render

    pic.render()