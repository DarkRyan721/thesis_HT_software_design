# ──────────────────────────────────────────────
# 🧰 Librerías estándar
import os
import numpy as np

# ──────────────────────────────────────────────
# 🧱 Qt (PySide6)
import PySide6.QtWidgets as QtW
from PySide6 import QtCore
from PySide6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox
)
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QThread, Signal, QObject, QTimer
# ──────────────────────────────────────────────
# 🌀 Visualización y mallas
import pyvista as pv
from pyvistaqt import QtInteractor
import meshio
import gmsh

# ──────────────────────────────────────────────
# 🧩 Módulos propios
from electric_field_solver import ElectricFieldSolver
from mesh_generator import HallThrusterMesh
from project_paths import model
from widgets.panels.simulation_options import SimulationOptionsPanel
from widgets.panels.magnetic_field.magnetic_options import MagneticOptionsPanel
from widgets.panels.field_options import FieldOptionsPanel
from gui_styles.stylesheets import *
from widgets.parameter_views import ParameterPanel
from widgets.options_panel import OptionsPanel
from widgets.view_panel import ViewPanel
from utils.loader_thread import LoaderWorker
from models.simulation_state import SimulationState
from widgets.panels.home_options import HomeOptionsPanel
from PySide6.QtWidgets import QProgressDialog
from PySide6.QtCore import QThread, Signal, QObject, Qt
from PySide6.QtGui import QGuiApplication

class MainWindow(QtW.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HET simulator")
        screen_size = QGuiApplication.primaryScreen().availableGeometry().size()
        w, h = screen_size.width(), screen_size.height()
        self.setMinimumSize(int(w * 0.8), int(h * 0.8))
        self.resize(int(w * 0.95), int(h * 0.8))

        # self.simulation_state = SimulationState()

        state_file = model("simulation_state.json")

        # Intenta cargar el estado existente, si no existe crea uno nuevo por defecto
        if os.path.exists(state_file):
            print("[INFO] Cargando estado de simulación desde JSON.")
            self.simulation_state = SimulationState.load_from_json(state_file)
        else:
            print("[INFO] No existe archivo de estado. Usando valores por defecto.")
            self.simulation_state = SimulationState()
            self.simulation_state.save_to_json(state_file)  # crea archivo inicial

        self.home_panel = HomeOptionsPanel(self)
        self.field_panel = FieldOptionsPanel(self)
        self.magnetic_panel = MagneticOptionsPanel(self)
        self.simulation_panel = SimulationOptionsPanel(self)
        self._setup_ui()
        self.setStyleSheet(self_Style())
        self.View_Part.add_viewer("magnetic", self.magnetic_panel.magnetic_viewer)

    def _setup_ui(self):
        #_____________________________________________________________________________________________
        #                   Ventana principal

        central_widget = QtW.QWidget() #Frame base de aplicacion Qt
        self.frame = QtW.QHBoxLayout()  #Frame principal. Contiene las tres columnas visibles
        self.frame.setSpacing(0) # Espaciado cero entre
        self.frame.setContentsMargins(0,0,0,0) # Las columnas no tienen margen

        #_____________________________________________________________________________________________
        #                   Creacion de las tres columnas principales

        self.Parameters = ParameterPanel(self)
        self.Options = OptionsPanel(self, self.Parameters.parameters_view)
        self.View_Part = ViewPanel(self)
        self.parameters_view = self.Parameters.parameters_view
        self.set_active_button(self.Options.tab_buttons[0])

        self.frame.addWidget(self.Options, stretch=0.3)
        self.frame.addWidget(self.Parameters, stretch=1)
        self.frame.addWidget(self.View_Part.view_stack, stretch=4)

        #_____________________________________________________________________________________________
        #                   Añadiendo las configuraciones al frame base y frame principal

        central_widget.setLayout(self.frame)
        self.setCentralWidget(central_widget)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'frame_content'):
            new_width = int(self.width())
            new_height = int(self.height())
            self.frame_content.setMinimumSize(new_width, new_height)

    @property
    def current_viewer(self):
        return self.View_Part.view_stack.currentWidget()

    def on_dynamic_tab_clicked(self, index, btn):
        self.parameters_view.setCurrentIndex(index)
        self.set_active_button(btn)
        if index == 0:
            self.View_Part.switch_view("mesh")
        elif index == 1:
            self.View_Part.switch_view("field")
            self.alert_if_field_or_magnetic_outdated()
        elif index == 2:
            self.View_Part.switch_view("magnetic")
            self.alert_if_field_or_magnetic_outdated()
        elif index == 3:
            self.View_Part.switch_view("simulation")
            self.alert_if_field_or_magnetic_outdated()


    def set_active_button(self, active_btn):
        for btn in self.Options.tab_buttons:
            if btn == active_btn:
                btn.setStyleSheet(button_activate_style())
            else:
                btn.setStyleSheet(button_options_style())

    def launch_worker(self, worker, finished_callback):
        thread = QThread()
        worker.moveToThread(thread)
        print(f"🚀 Lanzando worker en modo: {worker.mode}")
        # ──────────────────────────────
        # Conexiones
        thread.started.connect(worker.run)
        worker.finished.connect(finished_callback)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)

        def cleanup():
            print(f"🧹 Limpiando thread del worker: {worker.mode}")
            thread.wait()
            thread.deleteLater()

        thread.finished.connect(cleanup)

        # ──────────────────────────────
        thread.start()

    def alert_if_field_or_magnetic_outdated(self):
        state = self.simulation_state
        msg = ""
        if state.field_outdated and state.magnetic_outdated:
            msg = "⚠️ Debe actualizar el campo eléctrico y el campo magnético tras cambiar el dominio de simulación para obtener resultados correctos."
        elif state.field_outdated:
            msg = "⚠️ Debe actualizar el campo eléctrico tras cambiar el dominio de simulación."
        elif state.magnetic_outdated:
            msg = "⚠️ Debe actualizar el campo magnético tras cambiar el dominio de simulación."
        if msg:
            box = QMessageBox(self)
            box.setWindowTitle("Actualizar campos")
            box.setText(msg)
            box.setIcon(QMessageBox.Warning)
            box.setStyleSheet("""
                QMessageBox {
                    background-color: #121212; /* fondo negro */
                    color: #f5f5f5;           /* texto blanco */
                }
                QLabel {
                    color: #f5f5f5;
                }
                QPushButton {
                    background-color: #23272b;
                    color: #f5f5f5;
                    border: 1px solid #444;
                    border-radius: 6px;
                    padding: 6px 18px;
                }
                QPushButton:hover {
                    background-color: #32373a;
                }
            """)
            box.exec()








