import sys
import os

import numpy as np
import pyvista as pv

from pyvistaqt import QtInteractor

from utils.field_loader import FieldLoaderWorker

# Añadir ../../ (es decir, src/) al path para importar desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QLabel,
    QHBoxLayout, QPushButton, QCheckBox, QSlider, QComboBox
)
from PySide6.QtCore import Qt
import PySide6.QtWidgets as QtW

# ──────────────────────────────────────────────
# 🧩 Módulos propios
from Gen_Mallado import HallThrusterMesh
from E_field_solver import ElectricFieldSolver
from styles.stylesheets import *
from widgets.parameter_views import ParameterPanel
from widgets.options_panel import OptionsPanel
from widgets.view_panel import ViewPanel
from utils.mesh_loader import LoaderWorker
from utils.ui_helpers import _input_with_unit

class FieldOptionsPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.simulation_state = self.main_window.simulation_state
        self.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(self)

        # 🧮 Inputs
        self.input_Volt_container, self.input_Volt = _input_with_unit(str(self.simulation_state.voltage), "[V]")
        self.input_Volt_Cath_container, self.input_Volt_Cath = _input_with_unit(str(self.simulation_state.voltage_cathode), "[V]")

        self.field_viewer = self._create_viewer()

        field_box = QGroupBox("Electric Field Parameters")
        field_box.setSizePolicy(QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Fixed)

        sim_layout = QFormLayout()
        sim_layout.addRow("Voltaje (Volt):", self.input_Volt_container)
        sim_layout.addRow("Voltaje Cathode (Volt_Cath):", self.input_Volt_Cath_container)
        field_box.setLayout(sim_layout)
        layout.addWidget(field_box)

        # 🧩 Botón de actualización
        update_btn = QPushButton("Update")
        update_btn.setStyleSheet(button_parameters_style())
        update_btn.clicked.connect(self.on_update_clicked_Electric_field)
        layout.addWidget(update_btn)

        # 🎨 Panel de estilo visual
        layout.addStretch()
        self._load_initial_field_if_exists()

    def on_update_clicked_Electric_field(self):
        voltaje = float(self.input_Volt.text())
        voltaje_Cath = float(self.input_Volt_Cath.text())

        self.simulation_state.voltage = voltaje
        self.simulation_state.voltage_cathode = voltaje_Cath

        new_params = (voltaje, voltaje_Cath)
        if new_params != self.simulation_state.prev_params_field:
            print("🔄 ¡Parámetros cambiaron:", new_params)
            self.simulation_state.prev_params_field = new_params
            solver = ElectricFieldSolver()
            phi, E = solver.solve_laplace(Volt=voltaje, Volt_cath=voltaje_Cath)

            npy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data_files/Electric_Field_np.npy"))
            solver.save_electric_field_numpy(E, filename=npy_path)
                # Forzar visualización del viewer
            self.worker = LoaderWorker(mode="field")
            self.worker.finished.connect(self.on_field_loaded)
            self.worker.start()
        else:
            print("⚠️ No se han realizado cambios en los parámetros del campo.")

    def on_field_loaded(self, data):
        self.current_field = data
        if self.main_window.parameters_view.currentIndex() == 1:
            self.main_window.View_Part.switch_view("field")
        self.visualize_field(data)

    def visualize_field(self, data):
        self.field_viewer.clear()
        self.field_viewer.add_mesh(data, scalars="magnitude", cmap="plasma")
        self.field_viewer.reset_camera()

    def _create_viewer(self):
        viewer = QtInteractor()
        viewer.set_background("gray")
        viewer.setStyleSheet("background-color: #131313; border-radius: 5px;")
        viewer.add_axes(interactive=False)
        viewer.setSizePolicy(QtW.QSizePolicy.Expanding, QtW.QSizePolicy.Expanding)
        viewer.view_zx()
        return viewer

    def _load_initial_field_if_exists(self):
        field_path = "./data_files/Electric_Field_np.npy"
        if os.path.exists(field_path):
            print("⚡ Cargando campo eléctrico inicial...")
            self.worker = LoaderWorker(mode="field")
            self.worker.finished.connect(self.on_field_loaded)
            self.worker.start()