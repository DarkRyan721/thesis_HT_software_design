import sys
import os

import numpy as np
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

# ──────────────────────────────────────────────
# 🧩 Módulos propios
from Gen_Mallado import HallThrusterMesh
from E_field_solver import ElectricFieldSolver
from styles.stylesheets import *
from widgets.parameter_views import ParameterPanel
from widgets.options_panel import OptionsPanel
from widgets.view_panel import ViewPanel
from utils.mesh_loader import MeshLoaderWorker
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


    def on_update_clicked_Electric_field(self):
        voltaje = float(self.input_Volt.text())
        voltaje_Cath = float(self.input_Volt_Cath.text())

        self.simulation_state.voltage = voltaje
        self.simulation_state.voltage_cathode = voltaje_Cath

        print(f"voltaje = {voltaje}, voltaje_cath = {voltaje_Cath}")
        print(self.simulation_state.prev_params_field)
        new_params = (voltaje, voltaje_Cath)
        if new_params != self.simulation_state.prev_params_field:
            print("🔄 ¡Parámetros cambiaron:", new_params)
            self.simulation_state.prev_params_field = new_params
            solver_electric_field = ElectricFieldSolver()
        else:
            print("⚠️ No se han realizado cambios en la malla.")

    def add_field_vectors_from_npy(self, npy_path, scale=0.01):
        data = np.load(npy_path)
        if data.shape[1] < 6:
            raise ValueError("El archivo .npy debe contener [x, y, z, Ex, Ey, Ez]")

        points = data[:, :3]
        vectors = data[:, 3:]
        magnitudes = np.linalg.norm(vectors, axis=1)
        log_magnitudes = np.log10(magnitudes + 1e-3)

        mesh = pv.PolyData(points)
        mesh["vectors"] = vectors
        mesh["magnitude"] = log_magnitudes

        self.field_viewer.clear()
        self.field_viewer.add_mesh(mesh.glyph(orient="vectors", scale=False, factor=scale),
                            scalars="magnitude", cmap="plasma")
        self.field_viewer.add_axes(interactive=False)
        self.field_viewer.reset_camera()

    def _create_viewer(self):
        viewer = QtInteractor()
        viewer.set_background("gray")
        viewer.setStyleSheet("background-color: #131313; border-radius: 5px;")
        viewer.add_axes(interactive=False)
        viewer.setSizePolicy(QtW.QSizePolicy.Expanding, QtW.QSizePolicy.Expanding)
        return viewer