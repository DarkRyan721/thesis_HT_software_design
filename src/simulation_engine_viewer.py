import numpy as np
import pyvista as pv
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyvistaqt import QtInteractor

import os


class Simulation():
    def __init__(self, plotter, L=0.02, Rin=0.028, Rex=0.05):
        # data_dir = os.path.join(os.path.dirname(__file__), "../data_files")
        # self.particle_path = os.path.abspath(os.path.join(data_dir, "particle_simulation.npy"))
        self.particle_path = "particle_simulation.npy"

        """
            L -> Longitud/profundidad del propulsor
            Rin -> Radio interno
            Rex -> Radio externo

            all_positions -> Variable contiene la renderizacion de las particulas

        """
        #_____________________________________________________________________________________________________
        #       Inicializacion de variables geometricas
        self.plotter = plotter
        self.L = L
        self.Rin = Rin
        self.Rex = Rex

        #_____________________________________________________________________________________________________
        #       Cargar posiciones de las particulas en el tiempo

        self.all_positions = np.load(self.particle_path, mmap_mode="r")
        self.num_frames, self.num_particles, _ = self.all_positions.shape

        #_____________________________________________________________________________________________________
        #       Creacion de variables de control de simulacion

        self.window_closed = False # Variable que detecta si la ventana sigue abierta
        self.pause = {"valor": False} # Variable el estado de pausa

        #_____________________________________________________________________________________________________
        #       creacion geometrica del propulsor

        self.Geometries_creation()

        #_____________________________________________________________________________________________________
        #       configuracion de la camara

        self.Plot_Configuration()

        #_____________________________________________________________________________________________________

    def on_close(self):
        self.window_closed = True

    def pause_simulation(self):
        self.pause["valor"] = not self.pause["valor"]
        estado = "Pausada" if self.pause["valor"] else "Ranudada"
        print(f"\n⏸️  Simulación: {estado}")

    def Geometries_creation(self):
        """
            Geometries_creation:

            Funcion encargada de crear el propuslor en la simulacion.
        """

        #_____________________________________________________________________________________________________
        #       Creacion de variables geometricas necesarias

        Rsol_ext = self.Rin/2 # Radio de los solenoides externos
        ancho_plano = (self.Rex*2)+(self.Rin) # Ancho de las "tapas" del propulsor
        espesor_plano = 0.001

        helping_value_Rin = 0.01*self.Rin
        helping_value_L = 0.01*self.L

        centro = (0, 0, (self.L) / 2)
        centro_solenoid = self.Rex
        direccion = (0, 0, 1)

        #_____________________________________________________________________________________________________
        #       Creacion del cilindro o canal de aceleracion

        cilindro_ext = pv.Cylinder(center=centro, direction=direccion, radius=self.Rex, height=self.L, resolution=200).triangulate()
        cilindro_int = pv.Cylinder(center=centro, direction=direccion, radius=self.Rin, height=self.L, resolution=200).triangulate()

        cilindro_hueco = cilindro_ext.boolean_difference(cilindro_int).clean()
        cilindro = cilindro_hueco.clip(normal="z", origin=(centro[0], centro[1], self.L - 1e-6), invert=True)

        cilindro_tapa = pv.Cylinder(center=centro, direction=direccion, radius=self.Rin-helping_value_Rin, height=self.L+helping_value_L, resolution=200).triangulate()

        #_____________________________________________________________________________________________________
        #       Creacion de la tapa plana posterior

        plano_solid = pv.Cube(center=(0, 0, 0), x_length=ancho_plano, y_length=ancho_plano, z_length=espesor_plano).triangulate()

        #_____________________________________________________________________________________________________
        #       Creacion de la tapa frontal(tapa hueca)

        plano_hueco_aux = pv.Cube(center=(0, 0, self.L), x_length=ancho_plano, y_length=ancho_plano, z_length=espesor_plano).triangulate()
        cilindro_corte = pv.Cylinder(center=(0, 0, self.L), direction=direccion,radius=self.Rex,height=10,resolution=100).triangulate()

        plano_hueco = plano_hueco_aux.boolean_difference(cilindro_corte)

        #_____________________________________________________________________________________________________
        #       Creacion de los cilindros exteriores(carcasas de los solenoides exteriores)

        cilindro_1 = pv.Cylinder(center=(centro_solenoid, centro_solenoid, (self.L)/2), direction=(0, 0, 1), radius=Rsol_ext, height=self.L, resolution=50)
        cilindro_2 = pv.Cylinder(center=(centro_solenoid, -centro_solenoid, (self.L)/2), direction=(0, 0, 1), radius=Rsol_ext, height=self.L, resolution=50)
        cilindro_3 = pv.Cylinder(center=(-centro_solenoid, centro_solenoid, (self.L)/2), direction=(0, 0, 1), radius=Rsol_ext, height=self.L, resolution=50)
        cilindro_4 = pv.Cylinder(center=(-centro_solenoid, -centro_solenoid, (self.L)/2), direction=(0, 0, 1), radius=Rsol_ext, height=self.L, resolution=50)


        #_____________________________________________________________________________________________________
        #       Creacion del objeto plot y de cada una de las geometrias antes desarrolladas

        self.plotter.set_background("black")
        self.plotter.add_mesh(cilindro, color="#656565", opacity=1, show_edges=False, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.2)
        self.plotter.add_mesh(plano_solid, color="gray", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)

        self.plotter.add_mesh(plano_hueco, color="gray", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
        self.plotter.add_mesh(cilindro_tapa, color="gray", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
        self.plotter.add_mesh(cilindro_1, color="#CD7F32", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
        self.plotter.add_mesh(cilindro_2, color="#CD7F32", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
        self.plotter.add_mesh(cilindro_3, color="#CD7F32", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)
        self.plotter.add_mesh(cilindro_4, color="#CD7F32", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3)

        plano_final = pv.Cube(center=(0, 0, 0.2), x_length=ancho_plano, y_length=ancho_plano, z_length=espesor_plano).triangulate() #BORRAR
        self.plotter.add_mesh(plano_final, color="gray", opacity=1, specular=1.0, specular_power=30, diffuse=0.8, ambient=0.3) #BORRAR


        #_____________________________________________________________________________________________________

    def Plot_Configuration(self):
        """
            Plot_Configuration:

            Funcion encargada de configurar los parametros de la simulacion.
        """

        #_____________________________________________________________________________________________________
        #       Configurar cámara

        self.plotter.camera_position = [(-5*self.Rex, 2.5*self.Rex, 5*self.Rex), (0, 0, 0), (0, 1, 0)]
        self.plotter.camera.view_angle = 60

        #_____________________________________________________________________________________________________
        #       Iluminacion adicional

        light = pv.Light(position=(-5*self.Rex, -5*self.Rex, -7*self.Rex), focal_point=(0, 0, 0), intensity=1.5)
        self.plotter.add_light(light)

        #_____________________________________________________________________________________________________
        #       Creacion de arreglo de partículas inicial separadas por iones y neutrones

        frame_data = self.all_positions[0]

        mask_ions = frame_data[:, 3] == 1
        mask_neutrals = frame_data[:, 3] == 0

        ions_points = frame_data[mask_ions, :3]
        neutrals_points = frame_data[mask_neutrals, :3]

        self.ions = pv.PolyData(ions_points)
        self.neutrals = pv.PolyData(neutrals_points)

        #_____________________________________________________________________________________________________
        #       Creacion de iones y neutrones en el plotter

        if self.num_particles <= 4000:
            particle_size = 3.0
        if self.num_particles <= 10000:
            particle_size = 1.2
        elif self.num_particles <= 100000:
            particle_size = 0.8
        elif self.num_particles <= 1000000:
            particle_size = 0.5


        if self.ions.n_points > 0:
            self.ion_actor = self.plotter.add_mesh(self.ions, color='deepskyblue', point_size=particle_size, render_points_as_spheres=True, lighting=True, specular=0.9, diffuse=1, ambient=0.3)
        else:
            self.ion_actor = None  # No hay iones en el primer frame

        if self.neutrals.n_points > 0:
            self.neutral_actor = self.plotter.add_mesh(self.neutrals, color='red', point_size=particle_size, render_points_as_spheres=True, lighting=True, specular=0.9, diffuse=1, ambient=0.3)
        else:
            self.neutral_actor = None  # No hay neutrales en el primer frame

        #_____________________________________________________________________________________________________
        #       Otras configuracion: observador de cierre de ventana y titulo

        self.plotter.add_text("\nHall Effect Thruster", position="upper_edge", color='white')

        # Callback de cierre de ventana
        self.plotter.iren.add_observer("ExitEvent", lambda *_: self.on_close())

        #_____________________________________________________________________________________________________
        #       Creacion de ejes [X,Y,Z]

        x_line = pv.Line(pointa=(-100, 0, 0), pointb=(100, 0, 0))
        y_line = pv.Line(pointa=(0, -100, 0), pointb=(0, 100, 0))
        z_line = pv.Line(pointa=(0, 0, -100), pointb=(0, 0, 100))

        self.plotter.add_mesh(x_line, color='red', line_width=3)
        self.plotter.add_mesh(y_line, color='green', line_width=3)
        self.plotter.add_mesh(z_line, color='blue', line_width=3)

        #_____________________________________________________________________________________________________

    def Animation(self, neutral_visible = False):
        #_____________________________________________________________________________________________________
        #       Mostrar la ventana y asignacion de key events

        # self.plotter.show(auto_close=False, interactive_update=True)

        self.plotter.add_key_event("space", self.pause_simulation)

        #_____________________________________________________________________________________________________
        #       Creacion de buffers para las particulas del simulador

        max_particles = self.num_particles  # Tamaño máximo del buffer

        buffer_ions = np.full((max_particles, 3), np.nan, dtype=np.float32)
        buffer_neutrals = np.full((max_particles, 3), np.nan, dtype=np.float32)

        #_____________________________________________________________________________________________________
        #       Ciclo de trabajo para el renderizado

        if neutral_visible == False:
            self.neutral_actor.SetVisibility(False)

        for frame in range(self.num_frames):
            if self.window_closed:
                print("\n")
                break

            while self.pause["valor"]:
                self.plotter.update()
                time.sleep(0.001)
                frame -= 1

            frame_data = self.all_positions[frame]

            mask_ions = frame_data[:, 3] == 1

            ions_points = frame_data[mask_ions, :3]

            # Update ions
            num_ions = min(len(ions_points), max_particles)

            if self.ion_actor is not None:
                buffer_ions[:] = np.nan
                if num_ions > 0:
                    buffer_ions[:num_ions] = ions_points[:num_ions]
                    self.ion_actor.SetVisibility(True)
                    self.ions.points = buffer_ions
                    self.ion_actor.mapper.dataset.points = self.ions.points
                else:
                    self.ion_actor.SetVisibility(False)

                #self.ion_actor.SetVisibility(False)

            # Update neutrals
            if neutral_visible == True:
                mask_neutrals = frame_data[:, 3] == 0
                neutrals_points = frame_data[mask_neutrals, :3]
                num_neutrals = min(len(neutrals_points), max_particles)


                if self.neutral_actor is not None:
                    buffer_neutrals[:] = np.nan
                    if num_neutrals > 0:
                        buffer_neutrals[:num_neutrals] = neutrals_points[:num_neutrals]
                        self.neutral_actor.SetVisibility(True)
                        self.neutrals.points = buffer_neutrals
                        self.neutral_actor.mapper.dataset.points = self.neutrals.points
                    else:
                        self.neutral_actor.SetVisibility(False)
            else:
                num_neutrals = self.num_particles-num_ions

            self.plotter.update()
            time.sleep(1 / 60)
            print(f"\rFrame: {frame + 1}/{self.num_frames} | Iones: {num_ions} | Neutros: {num_neutrals}", end='', flush=True)

        print("\n")

        #_____________________________________________________________________________________________________

    def Plume_plane(self):
        # Cargar posiciones
        self.all_positions = np.load("data_files/particle_simulation.npy", mmap_mode="r")
        
        frame = 450
        frame_data = self.all_positions[frame]

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

# if __name__ == "__main__":
#     L = 0.02
#     Rext = 0.05
#     Rint = 0.028

#     simulacion = Simulation()

#     simulacion.Animation()
#     #simulacion.Plume_plane()