from dolfinx import fem, io

import sys
import os
from mpi4py import MPI

import numpy as np
import pyvista as pv

from pyvistaqt import QtInteractor

from density import ElectronDensityModel


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
from Gen_Mallado import HallThrusterMesh
from E_field_solver import ElectricFieldSolver
from styles.stylesheets import *
from widgets.parameter_views import ParameterPanel
from widgets.options_panel import OptionsPanel
from widgets.view_panel import ViewPanel
from utils.loader_thread import LoaderWorker
from utils.ui_helpers import _input_with_unit
from paths import data_file

class DensityOptionsPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.simulation_state = self.main_window.simulation_state
        self.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(self)
        self.thread = QThread()

        # Viewer
        os.chdir("data_files")
        xdmf_path = data_file("SimulationZone.xdmf")
        with io.XDMFFile(MPI.COMM_WORLD, xdmf_path, "r") as xdmf:
            domain = xdmf.read_mesh(name="SPT100_Simulation_Zone")
        self.density_viewer = QtInteractor(self)
        self.density_instance = ElectronDensityModel(plotter=self.density_viewer, domain= domain)

        # Botón de carga
        load_btn = QPushButton("Cargar Densidad")
        load_btn.setStyleSheet(button_parameters_style())
        load_btn.clicked.connect(self.on_load_density)
        layout.addWidget(load_btn)

        layout.addStretch()
        self._load_initial_density_if_exists()

    def _create_viewer(self):
        viewer = QtInteractor()
        viewer.set_background("gray")
        viewer.setStyleSheet("background-color: #131313; border-radius: 5px;")
        viewer.add_axes(interactive=False)
        viewer.setSizePolicy(QtW.QSizePolicy.Expanding, QtW.QSizePolicy.Expanding)
        viewer.view_zx()
        return viewer

    def _load_initial_density_if_exists(self):
        path = data_file("density_n0.npy")
        if os.path.exists(path):
            worker = LoaderWorker(mode="density")
            self.main_window.launch_worker(worker, self.on_density_loaded)

    def on_load_density(self):
        worker = LoaderWorker(mode="density", plotter=self.density_instance)
        self.main_window.launch_worker(worker, self.on_density_loaded)

    def visualize_density(self, data, log_min, log_max):
        self.density_viewer.clear()
        self.density_viewer.set_background("black")
        self.density_viewer.add_axes(color="white")
        self.density_viewer.add_mesh(
            data,
            scalars="n0_log",
            cmap="plasma",
            clim=[log_min, log_max],
            point_size=2,
            render_points_as_spheres=True,
            scalar_bar_args={
                'title': "ne [m⁻³] (log₁₀)\n",
                'color': 'white',
                'fmt': "%.1f",
            }
        )
        self.density_viewer.reset_camera()

    def on_density_loaded(self, data):
        self.current_density = data
        self.main_window.View_Part.current_data = data
        self.main_window.View_Part.switch_view("density")
        self.visualize_density(data["density"], data["log_min"], data["log_max"])


