import sys
import os
import numpy as np
from pyvistaqt import QtInteractor

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox
)
import PySide6.QtWidgets as QtW
from PySide6.QtCore import Qt

# Acceso al path raíz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# ───── Módulos propios ─────
from simulation_engine_viewer import Simulation
from utils.ui_helpers import _input_with_unit
from gui_styles.stylesheets import *
from utils.loader_thread import LoaderWorker


class SimulationOptionsPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.simulation_state = self.main_window.simulation_state
        self.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(self)

        # ▓▓▓ Grupo de parámetros básicos
        sim_box = QGroupBox("Parámetros de Simulación")
        sim_box.setStyleSheet(box_render_style())
        sim_layout = QFormLayout()

        self.input_N_particles_container, self.input_N_particles = _input_with_unit(str(self.simulation_state.N_particles), "[pasos]")
        self.input_frames_container, self.input_frames = _input_with_unit(str(self.simulation_state.frames), "[A]")

        self.simulation_viewer = QtInteractor(self)
        self.simulation_instance = Simulation(plotter=self.simulation_viewer)

        sim_layout.addRow("Número de particulas (N):", self.input_N_particles)
        sim_layout.addRow("Frames (N):", self.input_frames)
        sim_box.setLayout(sim_layout)
        layout.addWidget(sim_box)


        # ▓▓▓ Botón de ejecución
        exec_btn = QPushButton("Ejecutar Simulación")
        exec_btn.setStyleSheet(button_parameters_style())
        exec_btn.clicked.connect(self.on_run_simulation)
        layout.addWidget(exec_btn)

        layout.addStretch()

    def on_run_simulation(self):
        try:
            N_particles = int(self.input_N_particles.text())
            frames = int(self.input_frames.text())

            self.simulation_state.N_particles = N_particles
            self.simulation_state.frames = frames

            new_params = (N_particles, frames)

            if new_params != self.simulation_state.prev_params_simulation:
                print("🔄 Parámetros de simulación cambiaron:", new_params)
                self.simulation_state.prev_params_simulation = new_params

                # Si necesitas recalcular campos primero (ej. magnéticos), déjalo como magnetic
                # Pero si ya están listos, puedes correr directamente la simulación:
                self.loader_worker_simulation = LoaderWorker(
                    mode="simulation",
                    params={"neutral_visible": False},
                    plotter=self.simulation_instance
                )
                self.main_window.launch_worker(self.loader_worker_simulation, self.on_simulation_loaded)


            else:
                print("⚠️ No se han realizado cambios en los parámetros.")
                print(f"▶️ Ejecutando simulación con parámetros: {new_params}")
                self.loader_worker_simulation = LoaderWorker(
                    mode="simulation",
                    params={"neutral_visible": False},
                    plotter=self.simulation_instance
                )
                self.main_window.launch_worker(self.loader_worker_simulation, self.on_simulation_loaded)

        except ValueError:
            print("❌ Error: Parámetros inválidos")

    def on_simulation_finished(self, result):
        print("✅ Simulación finalizada:", result)

    def on_simulation_loaded(self, data):
        self.current_simulation = data
        self.main_window.View_Part.current_data = data  # asegúrate de sincronizar ViewPanel
        self.main_window.View_Part.switch_view("simulation")
        # self.visualize_simulation(data)

    def visualize_simulation(self, data):
        self.simulation_viewer.clear()
        self.simulation_viewer.add_mesh(data, scalars="magnitude", cmap="plasma")
        self.simulation_viewer.reset_camera()

    def _create_viewer(self):
        viewer = QtInteractor()
        viewer.set_background("gray")
        viewer.setStyleSheet("background-color: #131313; border-radius: 5px;")
        viewer.add_axes(interactive=False)
        viewer.setSizePolicy(QtW.QSizePolicy.Expanding, QtW.QSizePolicy.Expanding)
        viewer.view_zx()
        return viewer
