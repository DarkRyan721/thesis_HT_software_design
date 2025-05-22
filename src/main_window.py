# ──────────────────────────────────────────────
# 🧰 Librerías estándar
import numpy as np

# ──────────────────────────────────────────────
# 🧱 Qt (PySide6)
import PySide6.QtWidgets as QtW
from PySide6 import QtCore
from PySide6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox
)
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
from widgets.panels.simulation_options import SimulationOptionsPanel
from widgets.panels.density_options import DensityOptionsPanel
from widgets.panels.magnetic_options import MagneticOptionsPanel
from widgets.panels.field_options import FieldOptionsPanel
from styles.stylesheets import *
from widgets.parameter_views import ParameterPanel
from widgets.options_panel import OptionsPanel
from widgets.view_panel import ViewPanel
from utils.loader_thread import LoaderWorker
from models.simulation_state import SimulationState
from widgets.panels.home_options import HomeOptionsPanel
from PySide6.QtWidgets import QProgressDialog
from PySide6.QtCore import QThread, Signal, QObject, Qt

class MainWindow(QtW.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HET simulator")
        self.setMinimumSize(1500,800)
        self.resize(1500,800)

        self.simulation_state = SimulationState()
        self.home_panel = HomeOptionsPanel(self)
        print("panel 1")
        self.field_panel = FieldOptionsPanel(self)
        self.magnetic_panel = MagneticOptionsPanel(self)
        self.density_panel = DensityOptionsPanel(self)
        self.simulation_panel = SimulationOptionsPanel(self)
        self._setup_ui()
        self.frame.addWidget(self.Options, stretch=0.3)
        self.frame.addWidget(self.Parameters, stretch=1)
        self.frame.addWidget(self.View_Part.view_stack, stretch=2)
        self.setStyleSheet(self_Style())

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

        self.frame.addWidget(self.Parameters, stretch=1)
        self.frame.addWidget(self.Options, stretch=0.3)
        self.frame.addWidget(self.View_Part.view_stack, stretch=2)

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
        elif index == 2:
            self.View_Part.switch_view("magnetic")
        elif index == 3:
            self.View_Part.switch_view("density")
        elif index == 4:
            self.View_Part.switch_view("simulation")

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





