import sys
import os
import time
import numpy as np
from pyvistaqt import QtInteractor

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QLabel,
    QHBoxLayout, QPushButton, QCheckBox, QSlider, QComboBox
)
from PySide6.QtCore import QThread, Signal, QObject
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QStyle
import PySide6.QtWidgets as QtW
from PySide6.QtCore import Qt
from PySide6.QtCore import QTimer

from project_paths import data_file, worker
from widgets.panels.collapse import CollapsibleBox
from PySide6.QtWidgets import QVBoxLayout, QPushButton, QWidget, QLabel, QFrame

# Acceso al path raíz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# ───── Módulos propios ─────
from simulation_engine_viewer import Simulation
from utils.ui_helpers import _input_with_unit
from gui_styles.stylesheets import *
from utils.loader_thread import LoaderWorker
from PySide6.QtWidgets import QHBoxLayout

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


        self.simulation_running = False
        self.simulation_paused = False
        self.simulation_worker = LoaderSimulation(plotter=self.simulation_instance)
        self.simulation_thread = QThread()
        self.simulation_worker.moveToThread(self.simulation_thread)
        # 3. Conectar señales y slots
        self.simulation_thread.started.connect(self.simulation_worker.run)
        self.simulation_worker.finished.connect(self.simulation_thread.quit)
        self.simulation_worker.finished.connect(self.simulation_worker.deleteLater)
        self.simulation_thread.finished.connect(self.simulation_thread.deleteLater)
        self.loader_worker_simulation = None

        controls_layout = QHBoxLayout()

        self.start_btn = QPushButton("▶️ Simular")
        self.start_btn.setStyleSheet(button_parameters_style())
        self.start_btn.clicked.connect(self.on_run_simulation)
        controls_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("▶️ Continuar")
        self.pause_btn.setStyleSheet(button_parameters_style())
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self.on_pause_resume_clicked)
        controls_layout.addWidget(self.pause_btn)

        self.restart_btn = QPushButton("🔄 Restart")
        self.restart_btn.clicked.connect(self.on_restart_clicked)
        self.restart_btn.setStyleSheet(button_parameters_style())
        controls_layout.addWidget(self.restart_btn)

        layout.addLayout(controls_layout)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Gas Xenon", ""])
        self.view_mode_combo.setStyleSheet(box_render_style())
        layout.addWidget(self.view_mode_combo)


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

        output_box = QGroupBox("Salida de Simulación")
        output_box.setStyleSheet("""
            QGroupBox {
                background-color: #23272b;
                color: #f5f5f5;
                border: 1.5px solid #333;
                border-radius: 5px;
                margin-top: 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
            }
            QLabel {
                color: #f5f5f5;
                font-weight: normal;
                font-size: 13px;
                padding: 2px;
            }
        """)
        output_layout = QVBoxLayout(output_box)

        self.label_impulso = QLabel("Impulso específico: <b>---</b>")
        self.label_tiempo = QLabel("Tiempo de simulación: <b>---</b>")
        self.label_frames = QLabel("Número de frames: <b>---</b>")
        output_layout.addWidget(self.label_impulso)
        output_layout.addWidget(self.label_tiempo)
        output_layout.addWidget(self.label_frames)

        layout.addWidget(output_box)

        # --- Añade estos widgets a tu layout principal ---
        layout.addWidget(advanced_toggle)
        layout.addWidget(advanced_content)

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
                self.run_solver_in_subprocess()
            else:
                print("⚠️ No se han realizado cambios en los parámetros.")

        except ValueError:
            print("❌ Error: Parámetros inválidos")

    def run_solver_in_subprocess(self):
        import subprocess
        from PySide6.QtCore import QTimer

        N_particles = int(self.input_N_particles.text())
        frames = int(self.input_frames.text())

        args = [
            'python3', worker('particle_in_cell_process.py'),  # Ajusta el path si es necesario
            str(N_particles), str(frames),  # Pasa los parámetros necesarios
        ]
        # Aquí puedes implementar la lógica para ejecutar el solver en un subprocess
        # Por ejemplo, usando subprocess.run() o similar
        print("🔄 Ejecutando solver en subprocess...")
        # Simulación de ejecución
        self.solver_start_time = time.perf_counter()
        try:
            self.process = subprocess.Popen(args)
            print(f"[DEBUG] Proceso lanzado, PID: {self.process.pid}")
        except Exception as e:
            print(f"[ERROR] Fallo al lanzar el proceso: {e}")
            self.process = None
            return

        def check_output():
            if self.process is None:
                self.timer.stop()
                return
            if self.process.poll() is not None:
                self.timer.stop()
                elapsed = time.perf_counter() - self.solver_start_time
                print(f"[DEBUG] Proceso terminado correctamente. Tiempo total: {elapsed:.2f} s")
                self.process = None

                # Leer resultados desde JSON
                import json
                try:
                    with open("resultados_simulacion.json") as f:
                        resultados = json.load(f)
                    impulso = resultados.get("impulso_especifico", "N/A")
                except Exception as e:
                    print(f"[ERROR] No se pudieron leer los resultados: {e}")

                self.label_impulso.setText(f"Impulso específico: <b>{impulso} s</b>")
                self.label_tiempo.setText(f"Tiempo de simulación: <b>{elapsed} min</b>")
                self.label_frames.setText(f"Número de frames: <b>{frames}</b>")
        self.timer = QTimer()
        self.timer.timeout.connect(check_output)
        self.timer.start(300)

    def on_pause_resume_clicked(self):
        # Pausa/reanuda SOLO la animación visual
        if self.simulation_paused:
            self.simulation_paused = False
            self.pause_btn.setText("⏸️ Pausa")
            if hasattr(self, "loader_worker_simulation"):
                print("▶️ Animación reanudada.")
                self.simulation_thread.start()
                self.simulation_paused = False
        else:
            self.simulation_paused = True
            self.pause_btn.setText("▶️ Continuar")
            if hasattr(self, "loader_worker_simulation"):
                self.simulation_paused = True
                print("⏸️ Animación pausada.")
                # self.simulation_thread.pause()

    def on_restart_clicked(self):
        print("[DEBUG] Reiniciando simulación")
        if hasattr(self, 'loader_worker_simulation') and self.loader_worker_simulation is not None:
            self.loader_worker_simulation.stop()
        self.simulation_running = False
        self.simulation_paused = False
        # Limpia o reinicia la interfaz según necesites

class LoaderSimulation(QObject):
    finished = Signal(object)
    started = Signal()
    progress = Signal(int)

    def __init__(self, params=None, plotter=None):
        super().__init__()
        self.params = params or {}
        self.plotter = plotter
        self._is_running = True

    @Slot()
    def run(self):
        print("🚀 Iniciando simulación de partículas...")
        if self.plotter is None:
            print("⚠️ Instancia de simulación no proporcionada.")
            return
        try:
            print("🔄 Ejecutando simulación...")
            self.plotter.Animation(neutral_visible=self.params.get("neutral_visible", False))
            self.finished.emit("Simulation completed")
        except Exception as e:
            print(f"❌ Error durante la simulación: {e}")

    def pause(self):
        print("Pausing simulation in worker.")
        self._is_paused = True

    def resume(self):
        print("Resuming simulation in worker.")
        self._is_paused = False

    def stop(self):
        print("Stopping simulation in worker.")
        self._is_running = False
        self._is_paused = False