import sys
import os
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from magnetic_field_solver_cpu import B_Field
from utils.loader_thread import LoaderWorker

# Añadir ../../ (es decir, src/) al path para importar desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QPushButton
)
from PySide6.QtCore import QThread, Signal, QObject, QTimer

from PySide6.QtCore import Qt
import PySide6.QtWidgets as QtW

from utils.ui_helpers import _input_with_unit
from gui_styles.stylesheets import button_parameters_style
# TODO: Importar el solver y el loader apropiados para el campo magnético
# from E_field_solver import MagneticFieldSolver
# from utils.magnetic_loader import MagneticLoaderWorker
from project_paths import data_file


class MagneticOptionsPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.simulation_state = self.main_window.simulation_state
        self.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(self)
        self.thread = QThread()

        # Parámetros magnéticos
        self.input_nSteps_container, self.input_nSteps = _input_with_unit(str(self.simulation_state.nSteps), "[N_turns]")
        self.input_N_turns_container, self.input_N = _input_with_unit(str(self.simulation_state.N_turns), "[N_turns]")
        self.input_i_container, self.input_i = _input_with_unit(str(self.simulation_state.I), "[A]")

        # Visualizador 3D
        self.magnetic_viewer = self._create_viewer()

        mag_box = QGroupBox("Magnetic Field Parameters")
        mag_box.setSizePolicy(QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Fixed)
        form = QFormLayout()
        form.addRow("nSteps:", self.input_nSteps_container)
        form.addRow("Número de vueltas N_turns:", self.input_N_turns_container)
        form.addRow("Corriente i:", self.input_i_container)
        mag_box.setLayout(form)
        layout.addWidget(mag_box)

        # Botón de actualización
        update_btn = QPushButton("Update")
        update_btn.setStyleSheet(button_parameters_style())
        update_btn.clicked.connect(self.on_update_clicked_Magnetic_field)
        layout.addWidget(update_btn)

        # Estira para ocupar espacio
        layout.addStretch()

        # Carga inicial si existe
        self._load_initial_magnetic_if_exists()

    def on_update_clicked_Magnetic_field(self):
        nSteps = int(self.input_nSteps.text())
        N_turns = int(self.input_N.text())
        I = float(self.input_i.text())

        self.simulation_state.nSteps = nSteps
        self.simulation_state.N_turns = N_turns
        self.simulation_state.I = I

        new_params = (nSteps, N_turns, I)

        if new_params != self.simulation_state.prev_params_magnetic:
            print("🔄 Parámetros magnéticos cambiaron:", new_params)
            self.simulation_state.prev_params_magnetic = new_params

            worker = LoaderWorker(mode="magnetic", params=new_params)
            self.main_window.launch_worker(worker, self.on_magnetic_loaded)
        else:
            print("⚠️ No se han realizado cambios en los parámetros magnéticos.")

        self.simulation_state.print_state()

    def on_magnetic_loaded(self, data):
        self.current_magnetic = data
        self.main_window.View_Part.current_data = data  # asegúrate de sincronizar ViewPanel
        self.main_window.View_Part.switch_view("magnetic")
        self.visualize_magnetic(data)

    def visualize_magnetic(self, data):
        self.magnetic_viewer.clear()
        self.magnetic_viewer.add_mesh(data, scalars="magnitude", cmap="viridis")
        self.magnetic_viewer.reset_camera()

    def _create_viewer(self):
        viewer = QtInteractor()
        viewer.set_background("gray")
        viewer.setStyleSheet("background-color: #131313; border-radius: 5px;")
        viewer.add_axes(interactive=False)
        viewer.setSizePolicy(QtW.QSizePolicy.Expanding, QtW.QSizePolicy.Expanding)
        viewer.view_zx()
        return viewer

    def _load_initial_magnetic_if_exists(self):
        path = data_file("Magnetic_Field_np.npy")
        params = (
            self.simulation_state.nSteps,
            self.simulation_state.N_turns,
            self.simulation_state.I
        )

        if os.path.exists(path):
            worker = LoaderWorker(
                mode="magnetic",
                params=params,
                regenerate=False
            )
            self.main_window.launch_worker(worker, self.on_magnetic_loaded)
