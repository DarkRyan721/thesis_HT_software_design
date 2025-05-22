import sys
import os
import time

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
from PySide6.QtCore import QThread, Signal, QObject, QTimer

from PySide6.QtWidgets import QProgressDialog

import subprocess
import json
from PySide6.QtCore import QTimer

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
from paths import data_file, temp_data_file, project_file

class FieldOptionsPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.solver_instance = ElectricFieldSolver()
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

        self.charge_density_check = QCheckBox("Habilitar densidad de carga")
        self.charge_density_check.setChecked(False)  # Por defecto desactivado
        self.charge_density_check.setStyleSheet(checkbox_parameters_style())
        layout.addWidget(self.charge_density_check)

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
        validate_density = self.charge_density_check.isChecked()

        self.simulation_state.voltage = voltaje
        self.simulation_state.voltage_cathode = voltaje_Cath

        new_params = (voltaje, voltaje_Cath)

        regenerate = new_params != self.simulation_state.prev_params_field

        params = {
            "validate_density": validate_density,
            "voltage": voltaje,
            "voltage_cathode": voltaje_Cath
        }

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
            regenerate=regenerate,
        )
        print("Finalizando con la actualización")
        self.simulation_state.print_state()

        self.field_viewer.setEnabled(True)
        self.field_viewer.show()
        self.worker = None
        self.thread = None

    def visualize_field(self, data):
        self.field_viewer.clear()
        t0 = time.perf_counter()
        self.field_viewer.add_mesh(data, scalars="magnitude", cmap="plasma")
        t1 = time.perf_counter()
        print(f"🕒 Tiempo de renderizado: {t1 - t0:.2f} segundos")
        self.worker = None

    def _create_viewer(self):
        viewer = QtInteractor()
        viewer.set_background("gray")
        viewer.setStyleSheet("background-color: #131313; border-radius: 5px;")
        viewer.add_axes(interactive=False)
        viewer.setSizePolicy(QtW.QSizePolicy.Expanding, QtW.QSizePolicy.Expanding)
        viewer.view_zx()
        return viewer

    def _load_initial_field_if_exists(self):
        path = data_file("Electric_Field_np.npy")
        if os.path.exists(path):
            worker = LoaderWorker(
                mode="field",
                params={"validate_density": self.charge_density_check.isChecked()},
                regenerate=False,
                solver=self.solver_instance
            )
            self.main_window.launch_worker(worker, self.on_field_loaded)

    def run_solver_in_subprocess(self, finished_callback, params):
        import os
        import subprocess
        import json
        import time
        from PySide6.QtCore import QTimer

        run_solver_path = project_file("run_solver.py")
        input_file = temp_data_file("input.json")
        output_file = temp_data_file("output.npy")

        print(f"[DEBUG] Path de run_solver.py: {run_solver_path}")
        print(f"[DEBUG] Eliminando {output_file} si existe antes de lanzar el solver...")

        # Eliminar archivo antes de lanzar el proceso
        if os.path.exists(output_file):
            os.remove(output_file)

        with open(input_file, 'w') as f:
            json.dump(params, f)

        self.solver_start_time = time.perf_counter()
        try:
            print(f"[DEBUG] Intentando ejecutar: python3 {run_solver_path} {input_file} {output_file}")
            print(f"[DEBUG] working dir: {os.getcwd()}")
            print(f"[DEBUG] Existe run_solver.py?: {os.path.exists(run_solver_path)}")
            print(f"[DEBUG] Existe input_file?: {os.path.exists(input_file)}")

            self.process = subprocess.Popen(['python3', run_solver_path, input_file, output_file])
            print(f"[DEBUG] Proceso lanzado, PID: {self.process.pid}")
        except Exception as e:
            print(f"[ERROR] Fallo al lanzar el proceso: {e}")
            print(f"[ERROR] Intentando lanzar: python3 {run_solver_path} {input_file} {output_file}")
            self.process = None
            return

        self.solver_finished = False

        def check_output():
            print("[DEBUG] Esperando a que termine el solver...")
            if self.process is None:
                print("[ERROR] self.process es None; se detiene el timer.")
                self.timer.stop()
                return
            if self.process.poll() is not None and os.path.exists(output_file):
                print("[DEBUG] Proceso ha terminado y archivo de salida encontrado.")
                self.timer.stop()
                self.solver_finished = True
                elapsed = time.perf_counter() - self.solver_start_time
                print(f"[DEBUG] Proceso terminado correctamente. Tiempo total: {elapsed:.2f} s")
                print(f"[DEBUG] Flag solver_finished: {self.solver_finished}")
                data = np.load(output_file)
                self.process = None  # Añade esto para evitar futuras llamadas con proceso viejo
                self.load_field_with_worker(finished_callback, params)

        # Solo crear y arrancar el timer si el proceso existe
        self.timer = QTimer()
        self.timer.timeout.connect(check_output)
        self.timer.start(500)
        print("[DEBUG] Timer iniciado para verificar fin del proceso cada 1s.")

    def load_field_with_worker(self, finished_callback, params):
        """
        Lanza un LoaderWorker en modo 'field' para cargar y convertir el archivo .npy a PyVista.
        """
        worker = LoaderWorker(
            mode="field",
            params={
                "validate_density": params["validate_density"],  # O lo que corresponda
                "voltage": params["voltage"],
                "voltage_cathode": params["voltage_cathode"]
            },
            regenerate=False,  # Solo lectura, no regenerar
            solver=self.solver_instance
        )
        self.main_window.launch_worker(worker, finished_callback)

    def on_field_loaded(self, data):
        self.current_field = data
        if self.main_window.parameters_view.currentIndex() == 1:
            self.main_window.View_Part.switch_view("field")
        QTimer.singleShot(0, lambda: self.visualize_field(data))

        self.field_viewer.setEnabled(True)
        self.field_viewer.show()
        self.process = None

