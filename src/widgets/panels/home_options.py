import sys
import os
import numpy as np
import pyvista as pv

from pyvistaqt import QtInteractor

from utils.loading_bar import ProgressBarWidget
from widgets.panels.collapse import CollapsibleBox



# Añadir ../../ (es decir, src/) al path para importar desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QLabel,
    QHBoxLayout, QPushButton, QCheckBox, QSlider, QComboBox
)
from PySide6.QtCore import Qt
import PySide6.QtWidgets as QtW
from PySide6.QtCore import QThread, Signal, QObject, QTimer
# ──────────────────────────────────────────────
# 🧩 Módulos propios
from mesh_generator import HallThrusterMesh

from gui_styles.stylesheets import *
from widgets.parameter_views import ParameterPanel
from widgets.options_panel import OptionsPanel
from widgets.view_panel import ViewPanel
from utils.loader_thread import LoaderWorker
from utils.ui_helpers import _input_with_unit
import os
import subprocess
import json
import time
from PySide6.QtCore import QTimer

from PySide6.QtWidgets import QVBoxLayout, QPushButton, QWidget, QLabel, QFrame
from project_paths import data_file, model, project_file, temp_data_file, worker

class HomeOptionsPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.simulation_state = self.main_window.simulation_state
        self.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(self)

        # Inputs
        self.input_L_container, self.input_H = _input_with_unit(str(self.simulation_state.H), "[m]")
        self.input_R_Big_container, self.input_R_Big = _input_with_unit(str(self.simulation_state.R_big), "[m]")
        self.input_R_Small_container, self.input_R_Small = _input_with_unit(str(self.simulation_state.R_small), "[m]")

        self.home_viewer = self._create_viewer()

        sim_box = QGroupBox("Simulation domain")
        sim_layout = QFormLayout()
        sim_layout.addRow("Total length (H):", self.input_L_container)
        sim_layout.addRow("Big radius (R_big):", self.input_R_Big_container)
        sim_layout.addRow("Small radius (R_small):", self.input_R_Small_container)
        self.mesh_quality_box = QComboBox()
        self.mesh_quality_box.addItems(["Test", "Low", "Medium", "High", "Ultra"])
        self.mesh_quality_box.setStyleSheet(box_render_style())
        sim_layout.addRow(self.mesh_quality_box)
        sim_box.setLayout(sim_layout)
        layout.addWidget(sim_box)

        self.update_btn = QPushButton("Update")
        self.update_btn.setStyleSheet(button_parameters_style())
        self.update_btn.clicked.connect(self.on_update_clicked_mesh)

        layout.addWidget(self.update_btn)

        # Panel de estilo (antes en _create_style_panel)
        style_box = QGroupBox("Visual Style")
        style_layout = QVBoxLayout(style_box)

        self.combo_render_mode = QComboBox()
        self.combo_render_mode.addItems(["Surface", "Wireframe", "Points", "Surface with edges"])
        self.combo_render_mode.setStyleSheet(box_render_style())
        style_layout.addWidget(QLabel("Render Mode"))
        style_layout.addWidget(self.combo_render_mode)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["Isometric", "XY", "XZ", "YZ"])
        self.view_combo.setStyleSheet(box_render_style())
        self.view_combo.currentTextChanged.connect(self.change_view)

        self.combo = QComboBox()
        self.combo.addItems(["Vista 3D", "Plano YZ"])
        self.combo.setStyleSheet(box_render_style())
        self.combo.currentTextChanged.connect(self.switch_dataset)

        # Añadir ambos combos al layout horizontal
        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("View Mode:"))  # Etiqueta opcional
        combo_layout.addWidget(self.view_combo)
        combo_layout.addWidget(QLabel("Display:"))    # Otra etiqueta opcional, o elimínala si no la quieres
        combo_layout.addWidget(self.combo)

        style_layout.addLayout(combo_layout)  # Añade el layout horizontal al vertical del view_box

        apply_btn = QPushButton("Apply Style")
        apply_btn.setStyleSheet(button_parameters_style())
        apply_btn.clicked.connect(lambda: self._apply_visual_style_home(self.displayed_mesh))

        style_layout.addWidget(apply_btn)
        layout.addWidget(style_box)

        # Fila horizontal para ambos combobox


        # --- Opciones avanzadas desplegables ---
        advanced_toggle = QPushButton("Opciones avanzadas")
        advanced_toggle.setCheckable(True)
        advanced_toggle.setChecked(False)
        advanced_toggle.setStyleSheet(advanced_toggle_style())

        advanced_content = QFrame()
        advanced_content.setVisible(False)
        advanced_content.setStyleSheet(advanced_content_style())

        self.input_min_scale_container, self.input_min_scale = _input_with_unit(
            str(self.simulation_state.min_physics_scale), "[m]"
        )
        self.input_max_elements_container, self.input_max_elements = _input_with_unit(
            str(self.simulation_state.max_elements), ""
        )
        advanced_content_layout = QFormLayout(advanced_content)

        # Campos de formulario para parámetros avanzados
        advanced_content_layout.addRow("Minimum Physics Scale:", self.input_min_scale_container)
        advanced_content_layout.addRow("Max Elements:", self.input_max_elements_container)


        def toggle_advanced():
            advanced_content.setVisible(advanced_toggle.isChecked())

        advanced_toggle.clicked.connect(toggle_advanced)
        layout.addWidget(advanced_toggle)
        layout.addWidget(advanced_content)


        # Añadir al layout principal

        self.setLayout(layout)

        self.progress_bar = ProgressBarWidget("Cargando malla...")
        layout.addWidget(self.progress_bar)
        self.progress_bar.hide()

        # Combobox para alternar vistas

        self.current_mesh = None
        layout.addStretch()
        self._load_initial_mesh_if_exists()

    def _create_viewer(self):
        viewer = QtInteractor()
        viewer.set_background("white")          # Fondo blanco
        viewer.setStyleSheet("background-color: #131313; border-radius: 5px;")
        viewer.add_axes(interactive=False)
        viewer.setSizePolicy(QtW.QSizePolicy.Expanding, QtW.QSizePolicy.Expanding)
        viewer.view_yx()
        return viewer

    def _apply_visual_style_home(self, mesh_to_plot):

        if mesh_to_plot is None:
            print("⚠️ No hay malla cargada.")
            return

        if not isinstance(mesh_to_plot, pv.PolyData):
            print(f"❌ mesh_to_plot no es PolyData, es {type(mesh_to_plot)} y su valor es {mesh_to_plot}")
            return
        self.home_viewer.clear()
        self.home_viewer.enable_eye_dome_lighting()
        self.home_viewer.enable_anti_aliasing()
        # self.home_viewer.renderer.RemoveAllLights()  # Elimina luces anteriores
        self.home_viewer.add_light(pv.Light(light_type='headlight'))  # Añadir luz frontal
        mode = self.combo_render_mode.currentText()
        opacity = 1

        common_kwargs = dict(
            opacity=opacity,
            lighting=True,
            smooth_shading=True,
            ambient=0.3,
            diffuse=0.8,
            specular=0.5,
            specular_power=15,
            split_sharp_edges=True,  # Mejora la definición de aristas
            feature_angle=30,        # Ángulo para dividir aristas afiladas
            line_width=1.5,
        )

        mode = self.combo_render_mode.currentText()

        self.home_viewer.clear()
        self.home_viewer.enable_eye_dome_lighting()  # Mejora el contraste visual 3D
        self.home_viewer.enable_anti_aliasing()      # Suaviza los bordes

        if mode == "Surface":
            self.home_viewer.add_mesh(
                mesh_to_plot,
                style="surface",
                color="steelblue",
                show_edges=False,
                **common_kwargs
            )
        elif mode == "Surface with edges":
            self.home_viewer.add_mesh(
                mesh_to_plot,
                style="surface",
                show_edges=True,
                edge_color="black",
                color="#cccccc",
                **common_kwargs
            )
        elif mode == "Wireframe":
            self.home_viewer.add_mesh(
                mesh_to_plot,
                style="wireframe",
                color="black",
                line_width=1.5,
                lighting=False,
                smooth_shading=False,
                ambient=0.0,
                diffuse=0.0,
                specular=0.0,
                opacity=opacity
            )
        elif mode == "Points":
            self.home_viewer.add_mesh(
                mesh_to_plot,
                style="points",
                render_points_as_spheres=True,
                point_size=5,
                color="cyan",
                **common_kwargs
            )
        self.home_viewer.show_bounds(
            all_edges=True,
            location="outer",
            color="white",
            grid=True,
            show_xaxis=True,
            show_yaxis=True,
            show_zaxis=True,
            xlabel="X [m]",     # Aquí pones la unidad
            ylabel="Y [m]",     # Aquí pones la unidad
            zlabel="Z [m]",     # Aquí pones la unidad
            ticks='outside',
            use_2d=False,
            corner_factor=0.0,
        )

        self.home_viewer.reset_camera()

    def on_update_clicked_mesh(self):
        # Paso 1: Extrae los textos de los campos
        campos = {
            'H': self.input_H.text(),
            'R_big': self.input_R_Big.text(),
            'R_small': self.input_R_Small.text(),
            'min_physical_scale': self.input_min_scale.text(),
            'max_elements': self.input_max_elements.text(),
        }

        opcional = ['min_physical_scale', 'max_elements']

        try:
            valores = self.validar_numeros(campos, opcionales=opcional)
        except ValueError as e:
            from PySide6.QtWidgets import QMessageBox
            print(str(e))
            QMessageBox.critical(self, "Error de validación", str(e))
            return

        # Así puedes usarlos directamente:
        H = valores['H']                  # Siempre float
        R_big = valores['R_big']          # Siempre float
        R_small = valores['R_small']      # Siempre float
        min_physical_scale = valores['min_physical_scale']  # Puede ser float o None
        max_elements = valores['max_elements']              # Puede ser float o None

        self.simulation_state.H = H
        self.simulation_state.R_big = R_big
        self.simulation_state.R_small = R_small
        self.refinement_level = self.mesh_quality_box.currentText().lower()

        print(f"H = {H}, R_big = {R_big}, R_small = {R_small}")
        new_params = (
            H, R_big, R_small, self.mesh_quality_box.currentText().lower(),
            min_physical_scale, max_elements
        )

        if new_params != self.simulation_state.prev_params_mesh:
            self.update_btn.setEnabled(False)
            self.progress_bar.start("Cargando malla...")

            print("🔄 ¡Parámetros cambiaron:", new_params)
            self.simulation_state.prev_params_mesh = new_params
            params = {
                "H": H,
                "R_Big": R_big,
                "R_Small": R_small,
                "refinement_level": self.refinement_level,
                "min_physical_scale": min_physical_scale,
                "max_elements": max_elements
            }
            self.run_mesher_in_subprocess(self.on_mesh_loaded, params)
        else:
            print("⚠️ No se han realizado cambios en la malla.")

        self.simulation_state.print_state()
        self.simulation_state.save_to_json(model("simulation_state.json"))

    def validar_numeros(self, campos, opcionales=None):
        """
        Valida que los valores de entrada sean numéricos y no vacíos,
        excepto los opcionales, que pueden ser vacíos o 'None' y se asignan como None.
        """
        if opcionales is None:
            opcionales = set()
        else:
            opcionales = set(opcionales)

        resultados = {}
        for nombre, texto in campos.items():
            texto = str(texto).strip()
            if not texto or texto.lower() == "none":  # <-- Ajuste aquí
                if nombre in opcionales:
                    resultados[nombre] = None
                    continue
                else:
                    raise ValueError(f"El campo '{nombre}' es obligatorio y está vacío.")
            try:
                valor = float(texto)
            except ValueError:
                raise ValueError(f"El campo '{nombre}' debe ser un número válido. Valor recibido: '{texto}'")
            resultados[nombre] = valor
        return resultados

    def on_mesh_loaded(self, data):
        print("[DEBUG] Callback on_mesh_loaded ejecutado")
        print(f"[DEBUG] Data recibida: {type(data)}")

        if not isinstance(data, pv.PolyData):
            print("❌ Error: Data recibida no es PolyData, es", type(data))
            return

        self.update_btn.setEnabled(True)
        self.progress_bar.finish()
        self.current_mesh = data
        self.main_window.View_Part.current_data = data
        self.main_window.View_Part.switch_view("mesh")
        self.switch_dataset(self.combo.currentText())

    def _load_initial_mesh_if_exists(self):
        path = data_file("SimulationZone.msh")
        if os.path.exists(path):
            worker = LoaderWorker(mode="mesh")
            self.main_window.launch_worker(worker, self.on_mesh_loaded)

    def run_mesher_in_subprocess(self, finished_callback, params):

        run_mesher_path = worker("mesh_generator_process.py")
        H = params["H"]
        R_big = params["R_Big"]
        R_small = params["R_Small"]
        refinement_level = params["refinement_level"]
        min_physical_scale = params["min_physical_scale"]
        max_elements = params["max_elements"]

        # Guarda los parámetros de mallado en JSON

        args = [
            'python3', run_mesher_path,
            str(H),
            str(R_big),
            str(R_small),
            refinement_level,
            "" if min_physical_scale is None else str(min_physical_scale),
            "" if max_elements is None else str(max_elements),
        ]

        self.process = subprocess.Popen(args)
        self.mesher_start_time = time.perf_counter()
        self.mesher_finished = False

        # Usa un QTimer para verificar cuándo termina el subproceso

        def check_output():
            if self.process is None:
                self.timer.stop()
                return
            if self.process.poll() is not None:
                self.timer.stop()
                self.mesher_finished = True
                elapsed = time.perf_counter() - self.mesher_start_time
                print(f"[DEBUG] Mallado terminado en {elapsed:.2f} s")
                self.process = None
                # Llama al callback
                print("[INFO] Recargando malla con LoaderWorker luego de mallado.")
                loader = LoaderWorker(mode="mesh")
                self.main_window.launch_worker(loader, self.on_mesh_loaded)
                # (Opcional) llamar al callback adicional si quieres

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: check_output())
        self.timer.start(300)  # cada 300 ms

    def change_view(self, view):
        if view == "Isometric":
            self.home_viewer.view_isometric()
            pos, fp, viewup = self.home_viewer.camera_position
            # Invertir la dirección de la cámara respecto al foco (fp)
            inverted_pos = (2*fp[0] - pos[0], 2*fp[1] - pos[1], 2*fp[2] - pos[2])
            self.home_viewer.camera_position = [inverted_pos, fp, viewup]
            self.home_viewer.reset_camera()
        elif view == "XY":
            self.home_viewer.view_xy()
        elif view == "XZ":
            self.home_viewer.view_xz()
        elif view == "YZ":
            self.home_viewer.view_yz()


    def switch_dataset(self, view_name):

        if self.current_mesh is None:
            print("⚠️ No hay malla cargada para mostrar.")
            return

        if view_name == "Vista 3D":
            mesh_to_show = self.current_mesh
        elif view_name == "Plano YZ":
            mesh_to_show = self.current_mesh.slice(normal='x')
        else:
            return

        # VERIFICACIÓN:
        if not isinstance(mesh_to_show, pv.PolyData):
            print(f"❌ Error: mesh_to_show no es PolyData, es {type(mesh_to_show)}")
            return

        self.displayed_mesh = mesh_to_show
        if not isinstance(mesh_to_show, pv.PolyData):
            print(f"❌ Error: mesh_to_show no es PolyData, es {type(mesh_to_show)}")
            return
        
        self._apply_visual_style_home(mesh_to_show)

