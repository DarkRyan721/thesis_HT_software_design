import subprocess
import sys
import os
import time
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Slot

import hashlib
import json

from magnetic_field_solver_cpu import B_Field
from utils.loader_thread import LoaderWorker
from widgets.panels.collapse import CollapsibleBox
from widgets.panels.magnetic_field.field_lines_control_panel import FieldLinesControlPanel
from widgets.panels.magnetic_field.heatmap_points_control import HeatmapControlPanel
from widgets.panels.magnetic_field.solenoid_points_control import SolenoidPointsControlPanel

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QLabel,
    QHBoxLayout, QPushButton, QCheckBox, QSlider, QComboBox
)

# Añadir ../../ (es decir, src/) al path para importar desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QPushButton
)
from PySide6.QtCore import QThread, Signal, QObject, QTimer

from PySide6.QtCore import Qt
import PySide6.QtWidgets as QtW

from utils.ui_helpers import _input_with_unit
from gui_styles.stylesheets import *
# TODO: Importar el solver y el loader apropiados para el campo magnético
# from E_field_solver import MagneticFieldSolver
# from utils.magnetic_loader import MagneticLoaderWorker
from project_paths import data_file, worker

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QPushButton, QComboBox
)
from PySide6.QtWidgets import QStyle
from PySide6.QtWidgets import QVBoxLayout, QPushButton, QWidget, QLabel, QFrame
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel

class MagneticOptionsPanel(QWidget):
    plot_result_ready = Signal(object)
    def __init__(self, main_window):
        super().__init__()
        self._active_plot_threads = []
        self.main_window = main_window
        self.simulation_state = self.main_window.simulation_state
        self.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(self)
        self.thread = QThread()
        self.plot_result_ready.connect(self._on_plot_result_ready)

        # Parámetros magnéticos
        self.input_nSteps_container, self.input_nSteps = _input_with_unit(str(self.simulation_state.nSteps), "[N_turns]")
        self.input_N_turns_container, self.input_N = _input_with_unit(str(self.simulation_state.N_turns), "[N_turns]")
        self.input_i_container, self.input_i = _input_with_unit(str(self.simulation_state.I), "[A]")


        # --- Layout para visualizaciones ---
        self.visualization_layout = QVBoxLayout()
        layout.addLayout(self.visualization_layout)

        # --- Combo de selección de visualización ---
        self.visualization_selector = QComboBox()
        self.visualization_selector.addItem("3D Magnetic Field")
        self.visualization_selector.addItem("Heatmap")
        self.visualization_selector.addItem("Field Lines")
        self.visualization_selector.addItem("Solenoid Points")
        self.visualization_selector.currentIndexChanged.connect(self._switch_view)
        layout.addWidget(self.visualization_selector)

        # --- Diccionario de widgets de visualización ---
        self.visualization_widgets = {}

        # --- Viewer 3D ---
        self.magnetic_viewer = self._create_viewer()
        self.visualization_widgets["3D Magnetic Field"] = self.magnetic_viewer

        # --- Canvas Matplotlib (Heatmap) ---
        self.mpl_canvas = None  # Se crea bajo demanda


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

        self.solenoid_controls = SolenoidPointsControlPanel()
        self.solenoid_controls.btn_update.clicked.connect(self._show_solenoid_points_viewer)

        self.field_lines_controls = FieldLinesControlPanel()
        self.field_lines_controls.btn_update.clicked.connect(self._show_field_lines_viewer)

        self.heatmap_controls = HeatmapControlPanel()
        self.heatmap_controls.btn_update.clicked.connect(self._show_heatmap_viewer)

        layout.addWidget(self.solenoid_controls)
        layout.addWidget(self.field_lines_controls)
        layout.addWidget(self.heatmap_controls)


        # Estira para ocupar espacio
        layout.addStretch()

        # Carga inicial si existe
        self._load_initial_magnetic_if_exists()

    def on_update_clicked_Magnetic_field(self):
        # Recolecta campos obligatorios y avanzados
        campos = {
            "nSteps": self.input_nSteps.text(),
            "N_turns": self.input_N.text(),
            "I": self.input_i.text(),
        }

        try:
            valores = self.validar_numeros(campos)
        except ValueError as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error de validación", str(e))
            return

        nSteps = int(valores["nSteps"])
        N_turns = int(valores["N_turns"])
        I = valores["I"]
        self.simulation_state.nSteps = nSteps
        self.simulation_state.N_turns = N_turns
        self.simulation_state.I = I

        new_params = (nSteps, N_turns, I)
        regenerate = new_params != self.simulation_state.prev_params_magnetic
        if regenerate:
            print("🔄 Parámetros magnéticos cambiaron:", new_params)
            self.simulation_state.prev_params_magnetic = new_params
            self.run_bfield_external(nSteps, N_turns, I)
        else:
            worker = LoaderWorker(mode="magnetic", params=new_params)
            self.main_window.launch_worker(worker, self.on_magnetic_loaded)
            print("⚠️ No se han realizado cambios en los parámetros magnéticos.")

        self.simulation_state.print_state()

    def validar_numeros(self, campos, opcionales=None):
        """
        Valida que los valores de entrada sean numéricos y no vacíos,
        excepto los opcionales, que pueden ser vacíos o 'None' y se asignan como None.
        """
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

    def run_bfield_external(self, nSteps, N_turns, I):
        script_path = worker("magnetic_field_process.py")
        args = [
            "python3", script_path,
            str(nSteps), str(N_turns), str(I),
        ]
        self.process = subprocess.Popen(args)
        self.magnetic_start_time = time.perf_counter()
        self.magnetic_finished = False

        def check_output():
            if self.process is None:
                self.timer.stop()
                return
            if self.process.poll() is not None:
                self.timer.stop()
                self.magnetic_finished = True
                elapsed = time.perf_counter() - self.magnetic_start_time
                print(f"[DEBUG] Mallado terminado en {elapsed:.2f} s")
                self.process = None
                # Llama al callback
                print("[INFO] Recargando malla con LoaderWorker luego de mallado.")
                loader = LoaderWorker(mode="magnetic", params=(nSteps, N_turns, I))
                self.main_window.launch_worker(loader, self.on_magnetic_loaded)
                # (Opcional) llamar al callback adicional si quieres

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: check_output())
        self.timer.start(300)  # cada 300 ms

    def on_magnetic_loaded(self, data):
            self.current_magnetic = data
            self.main_window.View_Part.current_data = data  # asegúrate de sincronizar ViewPanel
            self.main_window.View_Part.switch_view("magnetic")
            self.visualize_magnetic(data)

    def visualize_magnetic(self, data):
        self.magnetic_viewer.clear()
        streamlines = data.streamlines(data, )
        self.magnetic_viewer.add_mesh(data, scalars="magnitude", cmap="viridis",scalar_bar_args={"title": "|B| [T]"})
        self.magnetic_viewer.reset_camera()
        self.magnetic_viewer.view_yx()

    def _create_viewer(self):
        viewer = QtInteractor()
        viewer.set_background("white")
        viewer.setStyleSheet("background-color: white; border-radius: 5px;")
        viewer.add_axes(interactive=False)
        viewer.setSizePolicy(QtW.QSizePolicy.Expanding, QtW.QSizePolicy.Expanding)
        viewer.view_zx()
        try:
            viewer.enable_shadows(False)
        except Exception:
            pass
        try:
            viewer.enable_anti_aliasing(False)
        except Exception:
            pass
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
            )
            self.main_window.launch_worker(worker, self.on_magnetic_loaded)

    def show_viewer3d(self):
        # Limpia el canvas matplotlib si está presente
        if hasattr(self, "mpl_canvas") and self.mpl_canvas is not None:
            if "magnetic_heatmap" in self.main_window.View_Part.viewers:
                self.main_window.View_Part.view_stack.removeWidget(self.mpl_canvas)
                del self.main_window.View_Part.viewers["magnetic_heatmap"]
            self.mpl_canvas.deleteLater()
            self.mpl_canvas = None
        self.visualization_selector.setCurrentText("3D Magnetic Field")
        self.main_window.View_Part.switch_view("magnetic")


    def show_matplotlib_figure(self, fig, name):
        # Elimina el canvas anterior (si existe)
        if hasattr(self, "mpl_canvas") and self.mpl_canvas is not None:
            if name in self.main_window.View_Part.viewers:
                self.main_window.View_Part.view_stack.removeWidget(self.mpl_canvas)
                del self.main_window.View_Part.viewers[name]
            self.mpl_canvas.deleteLater()
            self.mpl_canvas = None

        self.mpl_canvas = FigureCanvas(fig)
        self.main_window.View_Part.add_viewer(name, self.mpl_canvas)
        self.main_window.View_Part.switch_view(name)


    def _switch_view(self):
        selected = self.visualization_selector.currentText()
        if selected == "3D Magnetic Field":
            self.show_viewer3d()
        elif selected == "Heatmap":
            self._show_heatmap_viewer()
        elif selected == "Field Lines":
            self._show_field_lines_viewer()
        elif selected == "Solenoid Points":
            self._show_solenoid_points_viewer()


    def _start_plot_thread(self, plot_func, finished_callback):
        thread = QThread()
        worker = PlotWorker(plot_func)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(finished_callback)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)

        def cleanup():
            print("🧹 Limpiando thread del PlotWorker")
            self._active_plot_threads.append((thread, worker))
            thread.deleteLater()

        thread.finished.connect(cleanup)
        self._active_plot_threads.append((thread, worker))
        thread.start()
        print("[DEBUG] Hilo de plot iniciado")

    def on_heatmap_finished(self, result):
        self.plot_result_ready.emit(('Heatmap', result))
    def on_fieldlines_finished(self, result):
        self.plot_result_ready.emit(('Field Lines', result))
    def on_solenoidpoints_finished(self, result):
        self.plot_result_ready.emit(('Solenoid Points', result))

    def _on_plot_result_ready(self, data):
        tipo, result = data
        fig, ax = result
        # Aquí puedes mapear tipo a nombre de viewer si quieres
        if tipo == "Heatmap":
            name = "magnetic_heatmap"
        elif tipo == "Field Lines":
            name = "magnetic_fieldlines"
        elif tipo == "Solenoid Points":
            name = "solenoid_points"
        else:
            name = "unknown_plot"

        self.show_matplotlib_figure(fig, name)
        self.visualization_selector.setCurrentText(tipo)

    # def _show_heatmap_viewer(self):
    #     print("[DEBUG] Mostrando Heatmap Viewer")
    #     self.visualization_selector.setCurrentText("Heatmap")
    #     self.generate_heatmap_threaded()

    # def _show_field_lines_viewer(self):
    #     print("[DEBUG] Mostrando Field Lines Viewer")
    #     self.visualization_selector.setCurrentText("Field Lines")
    #     self.generate_fieldlines_threaded()

    # def _show_solenoid_points_viewer(self):
    #     print("[DEBUG] Mostrando Solenoid Points Viewer")
    #     self.visualization_selector.setCurrentText("Solenoid Points")
    #     self.generate_solenoidpoints_threaded()

    def _show_solenoid_points_viewer(self):
        print("[DEBUG] Mostrando Solenoid Points Viewer")
        self.visualization_selector.setCurrentText("Solenoid Points")
        # Limpia el viewer 3D de PyVista
        self.magnetic_viewer.clear()
        # Obtén los parámetros de los checkboxes o controles
        params = self.solenoid_controls.get_params()
        # Instancia el campo magnético según parámetros actuales
        bfield = B_Field(
            nSteps=self.simulation_state.nSteps,
            N=self.simulation_state.N_turns,
            I=self.simulation_state.I
        )
        # Llama al método PyVista, pasando el QtInteractor embebido
        bfield.Solenoid_points_plot_pyvista(**params, plotter=self.magnetic_viewer)
        self.magnetic_viewer.reset_camera()
        self.magnetic_viewer.view_isometric()

    def generate_heatmap_threaded(self):
        print("[DEBUG] Generando Heatmap en un hilo separado")
        params = self.heatmap_controls.get_params()
        bfield = B_Field(
            nSteps=self.simulation_state.nSteps,
            N=self.simulation_state.N_turns,
            I=self.simulation_state.I
        )
        print("[DEBUG] Params enviados a heatmap:", params)
        plot_func = lambda: bfield.B_Field_Heatmap(**params)
        self._start_plot_thread(plot_func, self.on_heatmap_finished)

    def generate_fieldlines_threaded(self):
        print("[DEBUG] Generando Field Lines en un hilo separado")
        params = self.field_lines_controls.get_params()
        bfield = B_Field(
            nSteps=self.simulation_state.nSteps,
            N=self.simulation_state.N_turns,
            I=self.simulation_state.I
        )
        print("[DEBUG] Params enviados a field lines:", params)
        plot_func = lambda: bfield.B_Field_Lines(**params)
        self._start_plot_thread(plot_func, self.on_fieldlines_finished)

    def generate_solenoidpoints_threaded(self):
        print("[DEBUG] Generando Solenoid Points en un hilo separado")
        params = self.solenoid_controls.get_params()
        bfield = B_Field(
            nSteps=self.simulation_state.nSteps,
            N=self.simulation_state.N_turns,
            I=self.simulation_state.I
        )
        print("[DEBUG] Params enviados a solenoid points:", params)
        plot_func = lambda: bfield.Solenoid_points_plot(**params)
        self._start_plot_thread(plot_func, self.on_solenoidpoints_finished)


    def params_hash(self, mode, nSteps, N_turns, I, kwargs):
        string = f"{mode}_{nSteps}_{N_turns}_{I}_{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(string.encode()).hexdigest()

    def generate_magnetic_image(self, mode, nSteps, N_turns, I, kwargs, output_path):
        args = [
            sys.executable,
            worker("magnetic_graphic_process.py"),
            mode,
            str(nSteps),
            str(N_turns),
            str(I),
            output_path,
            json.dumps(kwargs)
        ]
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR en generación gráfica:", result.stderr)
            return False
        print("OK: Imagen generada:", output_path)
        return True

    def _show_heatmap_viewer(self):
        print("[DEBUG] Mostrando Heatmap Viewer")
        self.visualization_selector.setCurrentText("Heatmap")
        params = self.heatmap_controls.get_params()
        nSteps = self.simulation_state.nSteps
        N_turns = self.simulation_state.N_turns
        I = self.simulation_state.I
        mode = "heatmap"
        hashname = self.params_hash(mode, nSteps, N_turns, I, params)
        output_path = os.path.join(data_file(""), f"heatmap_{hashname}.png")

        if not os.path.exists(output_path):
            # Lanza el proceso, preferiblemente en un QThread (ejemplo aquí sin QThread)
            self.generate_magnetic_image( mode, nSteps, N_turns, I, params, output_path)
        # Ahora muestra la imagen
        self._show_image_in_panel(output_path, "magnetic_heatmap")

    def _show_field_lines_viewer(self):
        print("[DEBUG] Mostrando Field Lines Viewer")
        self.visualization_selector.setCurrentText("Field Lines")
        params = self.field_lines_controls.get_params()
        nSteps = self.simulation_state.nSteps
        N_turns = self.simulation_state.N_turns
        I = self.simulation_state.I
        mode = "fieldlines"
        hashname = self.params_hash(mode, nSteps, N_turns, I, params)
        output_path = os.path.join(data_file(""), f"fieldlines_{hashname}.png")

        if not os.path.exists(output_path):
            # Lanza el proceso bloqueante (puedes migrar a QThread luego)
            self.generate_magnetic_image(mode, nSteps, N_turns, I, params, output_path)
        # Mostrar la imagen en el panel
        self._show_image_in_panel(output_path, "magnetic_fieldlines")

    def _show_image_in_panel(self, image_path, name):
        # Borra canvas anterior
        if hasattr(self, "mpl_canvas") and self.mpl_canvas is not None:
            if name in self.main_window.View_Part.viewers:
                self.main_window.View_Part.view_stack.removeWidget(self.mpl_canvas)
                del self.main_window.View_Part.viewers[name]
            self.mpl_canvas.deleteLater()
            self.mpl_canvas = None

        # Carga la imagen en QLabel/QPixmap
        label = QLabel()
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(True)  # Si quieres que se adapte al tamaño
        self.main_window.View_Part.add_viewer(name, label)
        self.main_window.View_Part.switch_view(name)

class PlotWorker(QObject):
    finished = Signal(object)  # Emite el resultado (fig, ax) o solo fig

    def __init__(self, plot_func, *args, **kwargs):
        super().__init__()
        self.plot_func = plot_func
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        print("[DEBUG] INICIO de plot_func en hilo secundario...")
        start = time.perf_counter()
        result = self.plot_func(*self.args, **self.kwargs)
        print(f"[DEBUG] FIN de plot_func, duró {time.perf_counter()-start:.2f} s")
        self.finished.emit(result)
        print("[DEBUG] Ejecutando worker de plot...")


