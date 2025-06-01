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

        # Grupo: Parámetros básicos de simulación
        sim_box = QGroupBox("🧮 Parámetros de Simulación")
        sim_box.setStyleSheet(box_render_style())
        sim_layout = QFormLayout()

        lbl_n_particles = QLabel("Número de partículas (N):")
        lbl_n_particles.setToolTip("Cantidad total de partículas simuladas en el dominio.")
        self.input_N_particles_container, self.input_N_particles = _input_with_unit(
            str(self.simulation_state.N_particles), "[#]"
        )

        lbl_frames = QLabel("Frames de animación:")
        lbl_frames.setToolTip("Número de pasos de simulación/frames visualizados.")

        self.input_frames_container, self.input_frames = _input_with_unit(
            str(self.simulation_state.frames), "[#]"
        )

        sim_layout.addRow(lbl_n_particles, self.input_N_particles)
        sim_layout.addRow(lbl_frames, self.input_frames)

        # Combo para tipo de gas
        lbl_gas = QLabel("Gas de trabajo:")
        lbl_gas.setToolTip("Selecciona el gas con el que se simula la descarga.")
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems([
            "Xenón (Xe) – estándar en propulsión",
            "Argón (Ar) – experimental, menor masa",
            "Helio (He) – alta movilidad, muy ligero"
        ])
        self.view_mode_combo.setStyleSheet(box_render_style())
        sim_layout.addRow(lbl_gas, self.view_mode_combo)

        sim_box.setLayout(sim_layout)
        layout.addWidget(sim_box)

        # ---- Botones de control de simulación ----
        controls_box = QGroupBox("Controles de simulación")
        controls_box.setStyleSheet(box_render_style())
        controls_layout = QHBoxLayout()

        self.start_btn = QPushButton("▶️ Simular")
        self.start_btn.setStyleSheet(button_parameters_style())
        self.start_btn.setToolTip("Inicia o reanuda la simulación de partículas.")
        self.start_btn.clicked.connect(self.on_run_simulation)
        controls_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("⏸️ Run")
        self.pause_btn.setStyleSheet(button_parameters_style())
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self.on_pause_resume_clicked)
        controls_layout.addWidget(self.pause_btn)

        # ----------- Aquí está el nuevo checkbox ------------
        self.enable_checkbox = QCheckBox("Habilitar computacion grafica")
        self.enable_checkbox.setChecked(True)  # o False según lo que desees por defecto
        self.enable_checkbox.setToolTip("Activa o desactiva el uso de CUDA (Nvidia).")
        controls_layout.addWidget(self.enable_checkbox)
        # Puedes conectar a una función si quieres hacer algo cuando cambie el estado:
        # -----------------------------------------------------

        controls_box.setLayout(controls_layout)
        layout.addWidget(controls_box)

        # ---- Opciones avanzadas ----
        advanced_toggle = QPushButton("Opciones avanzadas")
        advanced_toggle.setCheckable(True)
        advanced_toggle.setChecked(False)
        advanced_toggle.setStyleSheet(advanced_toggle_style())

        advanced_content = QFrame()
        advanced_content.setVisible(False)
        advanced_content.setStyleSheet(advanced_content_style())
        advanced_content_layout = QFormLayout(advanced_content)

        self.input_alpha_container, self.input_alpha = _input_with_unit(
            str(self.simulation_state.alpha), "[adim]"
        )
        self.input_alpha_container.setToolTip("Coeficiente de recombinación/adimensional para el modelo.")

        self.input_sigma_ion_container, self.input_sigma_ion = _input_with_unit(
            str(self.simulation_state.sigma_ion), "[m²]"
        )
        self.input_sigma_ion_container.setToolTip("Sección eficaz de ionización (m²).")

        self.input_dt_container, self.input_dt = _input_with_unit(
            str(self.simulation_state.dt), "[s]"
        )
        self.input_dt_container.setToolTip("Paso temporal de integración.")

        advanced_content_layout.addRow("Alpha:", self.input_alpha_container)
        advanced_content_layout.addRow("Sigma ion:", self.input_sigma_ion_container)
        advanced_content_layout.addRow("Delta time:", self.input_dt_container)

        def toggle_advanced():
            advanced_content.setVisible(advanced_toggle.isChecked())

        advanced_toggle.clicked.connect(toggle_advanced)
        layout.addWidget(advanced_toggle)
        layout.addWidget(advanced_content)


        # ---- Salida de la simulación ----
        output_box = QGroupBox("📊 Resultados de Simulación")
        output_box.setStyleSheet(box_render_style())
        output_layout = QVBoxLayout(output_box)
        self.label_impulso = QLabel("Impulso específico: <b>---</b>")
        self.label_tiempo = QLabel("Tiempo de simulación: <b>---</b>")
        self.label_frames = QLabel("Número de frames: <b>---</b>")
        output_layout.addWidget(self.label_impulso)
        output_layout.addWidget(self.label_tiempo)
        output_layout.addWidget(self.label_frames)
        layout.addWidget(output_box)

        layout.addStretch()
        self.setLayout(layout)

        # Inicialización de variables de simulación (como antes)
        self.simulation_viewer = QtInteractor(self)
        self.simulation_instance = Simulation(plotter=self.simulation_viewer)
        self.simulation_running = False
        self.simulation_paused = False
        self.simulation_worker = LoaderSimulation(plotter=self.simulation_instance)
        self.simulation_thread = QThread()
        self.simulation_worker.moveToThread(self.simulation_thread)
        self.simulation_thread.started.connect(self.simulation_worker.run)
        self.simulation_worker.finished.connect(self.simulation_thread.quit)
        self.simulation_worker.finished.connect(self.simulation_worker.deleteLater)
        self.simulation_thread.finished.connect(self.simulation_thread.deleteLater)
        self.loader_worker_simulation = None

    def on_run_simulation(self):
        campos = {
            "N_particles": self.input_N_particles.text(),
            "frames": self.input_frames.text(),
            "alpha": self.input_alpha.text(),
            "sigma_ion": self.input_sigma_ion.text(),
            "dt": self.input_dt.text(),
        }
        opcionales = ["alpha", "sigma_ion", "dt"]

        try:
            valores = self.validar_numeros(campos, opcionales=opcionales)
        except ValueError as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error de validación", str(e))
            return

        N_particles = int(valores["N_particles"])
        frames = int(valores["frames"])
        alpha = valores["alpha"]      # Puede ser float o None
        sigma_ion = valores["sigma_ion"]
        dt = valores["dt"]

        self.simulation_state.N_particles = N_particles
        self.simulation_state.frames = frames
        self.simulation_state.alpha = alpha
        self.simulation_state.sigma_ion = sigma_ion
        self.simulation_state.dt = dt

        gas_combo_index = self.view_mode_combo.currentIndex()
        gas_combo_text = self.view_mode_combo.currentText()
        # Opcional: Normaliza el valor para enviarlo limpio
        if "Xenón" in gas_combo_text:
            gas = "Xenon"
        elif "Argón" in gas_combo_text:
            gas = "Argon"
        elif "Helio" in gas_combo_text:
            gas = "Helium"
        elif "Kriptón" in gas_combo_text or "Krypton" in gas_combo_text:
            gas = "Krypton"
        else:
            gas = "Xenon"  # Por defecto

        # O si quieres, usa un diccionario de mapeo
        gas_map = {
            0: "Xenon",
            1: "Argon",
            2: "Helium",
            3: "Krypton"
        }
        gas = gas_map.get(gas_combo_index, "Xenon")

        new_params = (N_particles, frames, alpha, sigma_ion, dt, gas)

        if new_params != self.simulation_state.prev_params_simulation:
            print("🔄 Parámetros de simulación cambiaron:", new_params)
            self.simulation_state.prev_params_simulation = new_params
            self.run_solver_in_subprocess(N_particles, frames, alpha, sigma_ion, dt, gas)
        else:
            print("⚠️ No se han realizado cambios en los parámetros.")

    def validar_numeros(self, campos, opcionales=None):
        if opcionales is None:
            opcionales = set()
        else:
            opcionales = set(opcionales)
        resultados = {}
        for nombre, texto in campos.items():
            texto = str(texto).strip()
            if not texto or texto.lower() == "none":
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

    def run_solver_in_subprocess(self, N_particles, frames, alpha, sigma_ion, dt, gas):
        import subprocess
        from PySide6.QtCore import QTimer

        
        args = [
            'python3', worker("particle_in_cell_process.py"),
            str(N_particles),
            str(frames),
            "" if alpha is None else str(alpha),
            "" if sigma_ion is None else str(sigma_ion),
            "" if dt is None else str(dt),
            str(self.enable_checkbox.isChecked()),
            gas  # Nuevo argumento: el tipo de gas seleccionado
        ]
        print("🔄 Ejecutando solver en subprocess...")
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

                import json
                try:
                    with open("resultados_simulacion.json") as f:
                        resultados = json.load(f)
                    impulso = resultados.get("impulso_especifico", "N/A")
                except Exception as e:
                    print(f"[ERROR] No se pudieron leer los resultados: {e}")
                    impulso = "N/A"

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