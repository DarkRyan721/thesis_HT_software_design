import sys
import os
import pyvista as pv

from pyvistaqt import QtInteractor



# Añadir ../../ (es decir, src/) al path para importar desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QLabel,
    QHBoxLayout, QPushButton, QCheckBox, QSlider, QComboBox
)
from PySide6.QtCore import Qt
import PySide6.QtWidgets as QtW
from PySide6.QtCore import QTimer

# ──────────────────────────────────────────────
# 🧩 Módulos propios
from Gen_Mallado import HallThrusterMesh

from styles.stylesheets import *
from widgets.parameter_views import ParameterPanel
from widgets.options_panel import OptionsPanel
from widgets.view_panel import ViewPanel
from utils.mesh_loader import MeshLoaderWorker
from utils.ui_helpers import _input_with_unit

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
        sim_box.setLayout(sim_layout)
        layout.addWidget(sim_box)

        update_btn = QPushButton("Update")
        update_btn.setStyleSheet(button_parameters_style())
        update_btn.clicked.connect(self.on_update_clicked_mesh)
        layout.addWidget(update_btn)

        # Panel de estilo (antes en _create_style_panel)
        style_box = QGroupBox("Visual Style")
        style_layout = QVBoxLayout(style_box)

        self.combo_render_mode = QComboBox()
        self.combo_render_mode.addItems(["Surface", "Wireframe", "Points", "Surface with edges"])
        self.combo_render_mode.setStyleSheet(box_render_style())
        style_layout.addWidget(QLabel("Render Mode"))
        style_layout.addWidget(self.combo_render_mode)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        style_layout.addWidget(QLabel("Opacity"))
        style_layout.addWidget(self.opacity_slider)

        apply_btn = QPushButton("Apply Style")
        apply_btn.setStyleSheet(button_parameters_style())
        apply_btn.clicked.connect(self._apply_visual_style_home)
        style_layout.addWidget(apply_btn)

        layout.addWidget(style_box)
        layout.addStretch()

    def _create_viewer(self):
        viewer = QtInteractor()
        viewer.set_background("gray")
        viewer.setStyleSheet("background-color: #131313; border-radius: 5px;")
        viewer.add_axes(interactive=False)
        viewer.setSizePolicy(QtW.QSizePolicy.Expanding, QtW.QSizePolicy.Expanding)
        return viewer

    def _apply_visual_style_home(self):
        if self.main_window.View_Part.current_view is None:
            print("⚠️ No hay malla cargada.")
            return

        self.current_mesh = self.main_window.View_Part.current_view
        if self.current_mesh is None:
            print("⚠️ No hay malla cargada.")
            return

                # Limpiar viewer
        self.home_viewer.clear()
        self.home_viewer.enable_eye_dome_lighting()
        self.home_viewer.enable_anti_aliasing()
        self.home_viewer.renderer.RemoveAllLights()  # Elimina luces anteriores
        self.home_viewer.add_light(pv.Light(light_type='headlight'))  # Añadir luz frontal
        mode = self.combo_render_mode.currentText()
        print(f"Render mode: {mode}")
        opacity = self.opacity_slider.value() / 100.0

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
        )

        mode = self.combo_render_mode.currentText()

        self.home_viewer.clear()
        self.home_viewer.enable_eye_dome_lighting()  # Mejora el contraste visual 3D
        self.home_viewer.enable_anti_aliasing()      # Suaviza los bordes

        if mode == "Surface":
            self.home_viewer.add_mesh(
                self.current_mesh,
                style="surface",
                color="steelblue",
                show_edges=False,
                **common_kwargs
            )
        elif mode == "Surface with edges":
            self.home_viewer.add_mesh(
                self.current_mesh,
                style="surface",
                show_edges=True,
                edge_color="black",
                color="#cccccc",
                **common_kwargs
            )
        elif mode == "Wireframe":
            self.home_viewer.add_mesh(
                self.current_mesh,
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
                self.current_mesh,
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
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            ticks='outside',
            use_2d=False,  # evita pixelado exagerado
            corner_factor=0.0,  # desactiva etiquetas redundantes en esquinas
        )

        self.home_viewer.reset_camera()

    def on_update_clicked_mesh(self):
        H = float(self.input_H.text())
        R_big = float(self.input_R_Big.text())
        R_small = float(self.input_R_Small.text())

        self.simulation_state.H = H
        self.simulation_state.R_big = R_big
        self.simulation_state.R_small = R_small

        print(f"H = {H}, R = {R_big}, R = {R_small}")
        print(self.simulation_state.prev_params_mesh)
        new_params = (H, R_big, R_small)
        if new_params != self.simulation_state.prev_params_mesh:
            print("🔄 ¡Parámetros cambiaron:", new_params)
            self.simulation_state.prev_params_mesh = new_params
            self.mesh_instance = HallThrusterMesh(R_big=R_big, R_small=R_small, H=H)
            self.mesh_instance.generate()
            self.worker = MeshLoaderWorker()
            self.worker.finished.connect(self.main_window.View_Part.update_mesh_viewer)
            self.worker.start()
        else:
            print("⚠️ No se han realizado cambios en la malla.")

    def on_mesh_loaded(self, mesh):
        self.main_window.View_Part.current_mesh = mesh
        self.main_window.View_Part.view_stack.setCurrentWidget(self.home_viewer)
        self._apply_visual_style_home(mesh)

    def _show_in_separate_window(self):

        mesh = self.current_mesh
        mesh.compute_normals(inplace=True)

        p = pv.Plotter()
        p.add_mesh(mesh, smooth_shading=True, color="steelblue", lighting=True)
        p.add_axes()
        p.show()