import sys
import os
import time

import numpy as np
import pyvista as pv

from pyvistaqt import QtInteractor

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

from PySide6.QtWidgets import QProgressDialog

import subprocess
import json
from PySide6.QtCore import QTimer

# ──────────────────────────────────────────────
# 🧩 Módulos propios
from mesh_generator import HallThrusterMesh
from electric_field_solver import ElectricFieldSolver
from gui_styles.stylesheets import *
from widgets.parameter_views import ParameterPanel
from widgets.options_panel import OptionsPanel
from widgets.view_panel import ViewPanel
from utils.loader_thread import LoaderWorker
from utils.ui_helpers import _input_with_unit
from project_paths import data_file, temp_data_file, project_file, worker
from PySide6.QtWidgets import QVBoxLayout, QPushButton, QWidget, QLabel, QFrame
from PySide6.QtWidgets import QVBoxLayout, QPushButton, QWidget, QLabel, QFrame

class FieldOptionsPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.solver_instance = ElectricFieldSolver()
        self.simulation_state = self.main_window.simulation_state
        self.setStyleSheet("background-color: transparent;")

        # Layout principal
        layout = QVBoxLayout(self)

        # Inputs voltaje
        self.input_Volt_container, self.input_Volt = _input_with_unit(str(self.simulation_state.voltage), "[V]")
        self.input_Volt_Cath_container, self.input_Volt_Cath = _input_with_unit(str(self.simulation_state.voltage_cathode), "[V]")

        # Viewer único
        self.field_viewer = self._create_viewer()

        # Almacenan los datos
        self.current_field = None
        self.current_density = None

        # Parámetros campo eléctrico
        field_box = QGroupBox("Electric Field Parameters")
        field_box.setSizePolicy(QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Fixed)
        sim_layout = QFormLayout()
        sim_layout.addRow("Voltaje (Volt):", self.input_Volt_container)
        sim_layout.addRow("Voltaje Cathode (Volt_Cath):", self.input_Volt_Cath_container)
        field_box.setLayout(sim_layout)
        layout.addWidget(field_box)

        # Check habilitar densidad
        self.charge_density_check = QCheckBox("Habilitar densidad de carga")
        self.charge_density_check.setChecked(False)
        self.charge_density_check.setStyleSheet(checkbox_parameters_style())
        layout.addWidget(self.charge_density_check)

        # Botón update
        update_btn = QPushButton("Update")
        update_btn.setStyleSheet(button_parameters_style())
        update_btn.clicked.connect(self.on_update_clicked_Electric_field)
        layout.addWidget(update_btn)

        # Toggle entre campo y densidad
        self.combo = QComboBox()
        self.combo.addItems(["Electric Field", "Electron Density"])
        self.combo.setStyleSheet(box_render_style())
        self.combo.currentTextChanged.connect(self.switch_dataset)
        layout.addWidget(self.combo)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["3D", "2D Slice ZY"])
        self.view_mode_combo.setStyleSheet(box_render_style())
        self.view_mode_combo.currentTextChanged.connect(self.update_view_mode)
        layout.addWidget(self.view_mode_combo)

        self.z_value_edit = QLineEdit()
        self.z_value_edit.setPlaceholderText("Z value (float)")
        self.z_value_edit.setEnabled(False)  # Solo habilitado en modo 2D
        self.z_value_edit.editingFinished.connect(self.update_z_value)
        layout.addWidget(QLabel("Z slice"))
        layout.addWidget(self.z_value_edit)
        self.current_z_value = 0.01


        self.current_z_index = 0
        self.z_values = None  # Aquí almacenarás los posibles valores de Z para el slider

        self.view_direction_combo = QComboBox()
        self.view_direction_combo.addItems([
            "Isometric",
            "XY (Top)",
            "ZY (Front)",
            "ZX (Side)",
            "XZ (Bottom)",
            "YX (Back)"
        ])
        self.view_direction_combo.setStyleSheet(box_render_style())
        self.view_direction_combo.currentTextChanged.connect(self.change_view_direction)
        layout.addWidget(self.view_direction_combo)

                    # Grupo de opciones avanzadas
        advanced_toggle = QPushButton("Opciones Avanzadas")
        advanced_toggle.setCheckable(True)
        advanced_toggle.setChecked(False)
        advanced_toggle.setStyleSheet("""
            QPushButton {
                background-color: #18191a;
                color: #f5f5f5;
                border: none;
                font-weight: bold;
                text-align: left;
                padding: 8px;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #23272b;
                color: #f5f5f5;
            }
        """)

        advanced_content = QFrame()
        advanced_content.setVisible(False)
        advanced_content.setStyleSheet("""
            QFrame {
                background-color: #23272b;
                color: #f5f5f5;
                border-radius: 3px;
                padding: 8px;
            }
        """)
        advanced_content_layout = QVBoxLayout(advanced_content)
        advanced_content_layout.addWidget(QLabel("Opciones avanzadas (label interno)"))

        def toggle_advanced():
            advanced_content.setVisible(advanced_toggle.isChecked())

        advanced_toggle.clicked.connect(toggle_advanced)

        # --- Añade estos widgets a tu layout principal ---
        layout.addWidget(advanced_toggle)
        layout.addWidget(advanced_content)

        # Viewer embebido
        layout.addWidget(self.field_viewer)
        layout.addStretch()
        self.setLayout(layout)

        # Carga inicial de datos si existen
        self._load_initial_field_if_exists()
        self._load_initial_density_if_exists()

    def update_view(self):
        if self.combo.currentText() == "Electric Field":
            self.visualize_field()
        elif self.combo.currentText() == "Electron Density":
            self.visualize_density()

    def _create_viewer(self):
        viewer = QtInteractor()
        viewer.set_background("gray")
        viewer.setStyleSheet("background-color: #131313; border-radius: 5px;")
        viewer.add_axes(interactive=False)
        viewer.setSizePolicy(QtW.QSizePolicy.Expanding, QtW.QSizePolicy.Expanding)
        viewer.view_zx()
        return viewer

    def on_update_clicked_Electric_field(self):
        voltaje = float(self.input_Volt.text())
        voltaje_Cath = float(self.input_Volt_Cath.text())
        validate_density = self.charge_density_check.isChecked()

        self.simulation_state.voltage = voltaje
        self.simulation_state.voltage_cathode = voltaje_Cath

        new_params = (voltaje, voltaje_Cath)
        regenerate = new_params != self.simulation_state.prev_params_field

        params = {
            "voltage": voltaje,
            "voltage_cathode": voltaje_Cath,
            "validate_density": validate_density
        }
        print("validate density:", validate_density)
        if regenerate:
            print("🔄 ¡Parámetros cambiaron:", new_params)
            self.simulation_state.prev_params_field = new_params
            self.run_solver_in_subprocess(self.on_field_loaded, params)
            return
        else:
            print("⚠️ No se han realizado cambios en los parámetros del campo.")
        self.loader_worker_field = LoaderWorker(
            mode="field",
            params=params,
        )
        self.main_window.launch_worker(self.loader_worker_field, self.on_field_loaded)
        print("Finalizando con la actualización")
        self.simulation_state.print_state()

        self.field_viewer.setEnabled(True)
        self.field_viewer.show()
        self.worker = None
        self.thread = None

    def on_field_loaded(self, data):
        self.current_field = data
        self.update_view()

    def on_density_loaded(self, data):
        self.current_density = data
        if self.view_mode_combo.currentText() == "2D Slice ZY":
            self.update_z_slider_range()
        self.update_view()

    def change_view_direction(self, text):
        # Elimina los métodos interactivos del viewer si hay alguno
        viewer = self.field_viewer
        # Puedes limpiar la cámara o ejes si lo deseas aquí

        if text == "Isometric":
            viewer.view_isometric()
        elif text == "XY (Top)":
            viewer.view_xy()
        elif text == "ZY (Front)":
            viewer.view_zy()
        elif text == "ZX (Side)":
            viewer.view_zx()
        elif text == "XZ (Bottom)":
            viewer.view_xz()
        elif text == "YX (Back)":
            viewer.view_yx()
        # Después de cambiar, puedes resetear la cámara si es necesario
        viewer.reset_camera()
    def visualize_field(self):
        if self.current_field is None:
            print("No hay datos de campo eléctrico para visualizar.")
            return

        self.field_viewer.clear()
        self.field_viewer.set_background("gray")
        self.field_viewer.add_axes(interactive=False)

        mode = self.view_mode_combo.currentText()
        if mode == "3D":
            # glyphs = self.current_field.glyph(orient="E_field", scale=False, factor=0.01)
            # self.field_viewer.add_mesh(glyphs, scalars="magnitude", cmap="plasma")
            glyphs = self.current_field.glyph(
                orient="E_field",       # nombre del vector en tu PolyData
                scale=False,            # no escalar por magnitud
                factor=0.01

            )
            self.field_viewer.add_mesh(
                glyphs,
                cmap="magma",                   # paleta académica
                scalars="magnitude",            # colorea por magnitud del campo
                clim=[
                    self.current_field["magnitude"].min(),
                    self.current_field["magnitude"].max()
                ],
                line_width=0.03,                 # flechas delgadas
                opacity=1,                    # ligeramente transparente
                show_scalar_bar=True,           # barra de colores (quita si no la quieres)
                render_points_as_spheres=False, # no muestres nodos como esferas
                lighting=True,
                scalar_bar_args={"title": "|Log10(E)| [V/m]"}

            )
            self.field_viewer.view_isometric()
            pos, fp, viewup = self.field_viewer.camera_position
            # Invertir la dirección de la cámara respecto al foco (fp)
            inverted_pos = (2*fp[0] - pos[0], 2*fp[1] - pos[1], 2*fp[2] - pos[2])
            self.field_viewer.camera_position = [inverted_pos, fp, viewup]
            self.field_viewer.reset_camera()
            self.field_viewer.add_text("Campo eléctrico (Vista isometrica)", position='upper_edge', font_size=12, color='black')
        elif mode == "2D Slice ZY":
            # ---- PROYECTAR EN EL PLANO X = x_plane ----
            x_plane = self.current_z_value
            tolerance = 0.008
            mesh = self.current_field
            mask = np.abs(mesh.points[:, 0] - x_plane) <= tolerance

            if not np.any(mask):
                self.field_viewer.add_text("Sin datos en este plano", color="black", font_size=16)
                return

            # Proyectar puntos al plano X=x_plane
            points = mesh.points[mask].copy()
            points[:, 0] = x_plane

            # Proyectar solo las componentes Z, Y del campo (poniendo X=0)
            if "E_field" not in mesh.point_data:
                self.field_viewer.add_text("No hay campo E_field en los datos", color="black", font_size=16)
                return
            vectors = mesh.point_data["E_field"][mask]
            vectors_proj = vectors.copy()
            vectors_proj[:, 0] = 0  # Quitar la componente X

            # PolyData con los puntos proyectados y vectores proyectados
            projected = pv.PolyData(points)
            projected["vectors"] = vectors_proj
            # Si quieres, también puedes proyectar el escalar de magnitud
            if "magnitude" in mesh.point_data:
                projected["magnitude"] = mesh.point_data["magnitude"][mask]

            # ---- GRAFICAR GLYPHS (FLECHAS) ----
            glyphs = projected.glyph(    orient="vectors",
            scale=False,
            factor=0.01)
            self.field_viewer.add_mesh(    glyphs,
            cmap="magma",            # Paleta neutra y académica
            scalars="magnitude",
            clim=[
                self.current_field["magnitude"].min(),
                self.current_field["magnitude"].max()
            ],
            line_width=0.45,            # Muy delgadas
            opacity=0.7,              # Suave, no saturado
            show_scalar_bar=True,      # Quita si no lo necesitas
            render_points_as_spheres=False,
            lighting=False,
            # scalar_bar_args={"title": "|E| [V/m]"})
            scalar_bar_args={"title": "|Log10(E)| [V/m]"})


            self.field_viewer.view_zy()
            self.field_viewer.add_text("Campo eléctrico (proyección ZY)", position='upper_edge', font_size=12, color='black')
        # Puedes agregar aquí el modo de líneas de campo (ver más abajo)
        self.field_viewer.reset_camera()

    def project_to_plane(mesh, z_plane=0.0, tolerance=None):
        points = mesh.points.copy()
        # Si quieres proyectar TODOS los puntos al plano Z
        points[:, 2] = z_plane

        # Si solo quieres proyectar puntos dentro de un rango de Z (por ejemplo +-0.01)
        if tolerance is not None:
            mask = np.abs(mesh.points[:, 2] - z_plane) <= tolerance
            points = points[mask]
            # Asegúrate de filtrar los datos asociados si usas esta opción (ver abajo)

        # Crear un nuevo mesh con los puntos proyectados (manteniendo los scalars y/o vectors)
        projected = pv.PolyData(points)
        # Copia scalars (ejemplo: 'n0_log')
        for name in mesh.point_data:
            data = mesh.point_data[name]
            if tolerance is not None:
                data = data[mask]
            projected.point_data[name] = data
        return projected

    def filter_and_project_to_plane_z(self, mesh, z_plane=0.0, tolerance=0.015):
        # Filtra solo los puntos cercanos al plano Z
        mask = np.abs(mesh.points[:, 2] - z_plane) <= tolerance
        if not np.any(mask):
            return None
        points = mesh.points[mask].copy()
        points[:, 2] = z_plane  # Proyecta a plano Z exacto

        projected = pv.PolyData(points)
        # Copia point_data (por ejemplo, 'n0_log' o 'magnitude')
        for name in mesh.point_data:
            projected.point_data[name] = mesh.point_data[name][mask]
        # Copia vectores si existen (ejemplo: 'vectors' o 'E_field')
        for name in mesh.point_data:
            if mesh.point_data[name].ndim == 2 and mesh.point_data[name].shape[1] == 3:
                projected.point_data[name] = mesh.point_data[name][mask]
        return projected

    def filter_and_project_to_plane_x(self, mesh, x_plane=0.0, tolerance=0.015):
        mask = np.abs(mesh.points[:, 0] - x_plane) <= tolerance
        if not np.any(mask):
            return None
        points = mesh.points[mask].copy()
        points[:, 0] = x_plane  # Proyecta todos al plano X exacto

        projected = pv.PolyData(points)
        for name in mesh.point_data:
            projected.point_data[name] = mesh.point_data[name][mask]
        return projected

    def visualize_density(self):
        if self.current_density is None:
            print("No hay datos de densidad para visualizar.")
            return

        dens = self.current_density
        self.field_viewer.clear()
        self.field_viewer.set_background("white")
        self.field_viewer.add_axes(color="white")

        mode = self.view_mode_combo.currentText()
        if mode == "3D":
            self.field_viewer.set_background("gray")
            self.field_viewer.add_axes(color="black")
            self.field_viewer.add_mesh(
                dens["density"],
                scalars="n0_log",
                cmap="plasma",
                clim=[dens["log_min"], dens["log_max"]],
                point_size=2,
                render_points_as_spheres=True,
                scalar_bar_args={
                    'title': "ne [m⁻³] (log₁₀)\n",
                    'color': 'black',
                    'fmt': "%.1f",
                }
            )
        elif mode == "2D Slice ZY":
            # slice_mesh = self.get_slice_at_z(dens["density"])
            # slice_mesh = self.project_to_plane(dens["density"], z_plane=self.current_z_value)
            slice_mesh = self.filter_and_project_to_plane_x(dens["density"], x_plane=self.current_z_value)
            if slice_mesh is not None and slice_mesh.n_points > 0:
                self.field_viewer.set_background("gray")
                self.field_viewer.add_axes(color="black")

                self.field_viewer.add_mesh(
                    slice_mesh,
                    scalars="n0_log",
                    cmap="plasma",
                    clim=[dens["log_min"], dens["log_max"]],
                    point_size=2,
                    render_points_as_spheres=True,
                    scalar_bar_args={
                        'title': "ne [m⁻³] (log₁₀)\n",
                        'color': 'black',
                        'fmt': "%.1f",
                    }
                )
                self.field_viewer.view_zy()
                self.field_viewer.add_text("Distribución de Densidad Electrónica", position='upper_edge', font_size=12, color='black')

            else:
                print("No hay puntos en el plano Z={} para la densidad.".format(self.current_z_value))
                # Opcional: Puedes mostrar un mensaje en el viewer o limpiar la vista
                self.field_viewer.clear()
                self.field_viewer.add_text(f"Sin datos en Z={self.current_z_value:.2f}", color="black", font_size=16)
        self.field_viewer.reset_camera()

    def update_view_mode(self, mode):
        if mode == "2D Slice ZY":
            self.z_value_edit.setEnabled(True)
        else:
            self.z_value_edit.setEnabled(False)
        self.update_view()

    def update_z_value(self):
        # Lee el valor directamente del QLineEdit
        try:
            z = float(self.z_value_edit.text())
        except Exception:
            z = 0.0  # Valor por defecto si hay error
        self.current_z_value = z
        self.update_view()

    def switch_dataset(self, view_name):
        self.update_view()

    def get_slice_at_z(self, mesh, tol=1e-6):
        z = self.current_z_value
        # Busca el valor Z más cercano si hay valores únicos
        try:
            zs = np.unique(mesh.points[:, 2])
            closest = zs[np.abs(zs - z).argmin()]
            if abs(closest - z) > tol:
                print(f"Aviso: el Z más cercano encontrado es {closest:.6f} (input={z:.6f})")
            z = closest
        except Exception:
            pass

        try:
            slice_mesh = mesh.slice(normal='x', origin=(0, 0, z))
            return slice_mesh
        except Exception as e:
            print(f"Error generando corte ZY en z={z}: {e}")
            return None
    def switch_dataset(self, view_name):
        if view_name == "Electric Field":
            self.visualize_field()
        elif view_name == "Electron Density":
            self.visualize_density()

    def _load_initial_field_if_exists(self):
        path = data_file("E_Field_Laplace.npy")
        if os.path.exists(path):
            worker = LoaderWorker(
                mode="field",
                params={"validate_density": self.charge_density_check.isChecked()},
                solver=self.solver_instance
            )
            self.main_window.launch_worker(worker, self.on_field_loaded)

    def _load_initial_density_if_exists(self):
        path = data_file("density_n0.npy")
        if os.path.exists(path):
            worker = LoaderWorker(mode="density")
            self.main_window.launch_worker(worker, self.on_density_loaded)

    def run_solver_in_subprocess(self, finished_callback, params):
        import subprocess
        from PySide6.QtCore import QTimer

        validate_density = int(self.charge_density_check.isChecked())
        volt = float(self.input_Volt.text())
        volt_cath = float(self.input_Volt_Cath.text())

        args = [
            'python3', worker('electric_field_process.py'),  # Ajusta el path si es necesario
            str(volt), str(volt_cath), str(validate_density)
        ]

        self.solver_start_time = time.perf_counter()
        try:
            self.process = subprocess.Popen(args)
            print(f"[DEBUG] Proceso lanzado, PID: {self.process.pid}")
        except Exception as e:
            print(f"[ERROR] Fallo al lanzar el proceso: {e}")
            self.process = None
            return

        self.solver_finished = False

        def check_output():
            if self.process is None:
                self.timer.stop()
                return
            if self.process.poll() is not None:
                self.timer.stop()
                self.solver_finished = True
                elapsed = time.perf_counter() - self.solver_start_time
                print(f"[DEBUG] Proceso terminado correctamente. Tiempo total: {elapsed:.2f} s")
                self.process = None
                self.load_field_with_worker(finished_callback, {
                    "validate_density": validate_density,
                    "voltage": volt,
                    "voltage_cathode": volt_cath
                })

        self.timer = QTimer()
        self.timer.timeout.connect(check_output)
        self.timer.start(300)

    def load_field_with_worker(self, finished_callback, params):
        worker = LoaderWorker(
            mode="field",
            params={
                "validate_density": params["validate_density"],
                "voltage": params["voltage"],
                "voltage_cathode": params["voltage_cathode"]
            },
            solver=self.solver_instance
        )
        self.main_window.launch_worker(worker, finished_callback)

