# ──────────────────────────────────────────────
# 🧰 Librerías estándar
import numpy as np

# ──────────────────────────────────────────────
# 🧱 Qt (PySide6)
import PySide6.QtWidgets as QtW
from PySide6 import QtCore
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton,
    QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox
)

# ──────────────────────────────────────────────
# 🌀 Visualización y mallas
import pyvista as pv
from pyvistaqt import QtInteractor
import meshio
import gmsh

# ──────────────────────────────────────────────
# 🧩 Módulos propios
from E_field_solver import ElectricFieldSolver
from Gen_Mallado import HallThrusterMesh
from widgets.panels.magnetic_options import MagneticOptionsPanel
from widgets.panels.field_options import FieldOptionsPanel
from styles.stylesheets import *
from widgets.parameter_views import ParameterPanel
from widgets.options_panel import OptionsPanel
from widgets.view_panel import ViewPanel
from utils.mesh_loader import LoaderWorker
from models.simulation_state import SimulationState
from widgets.panels.home_options import HomeOptionsPanel

class MainWindow(QtW.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HET simulator")
        self.setMinimumSize(800,400)
        self.resize(800,400)

        self.simulation_state = SimulationState()
        self.home_panel = HomeOptionsPanel(self)
        self.field_panel = FieldOptionsPanel(self)
        self.magnetic_panel = MagneticOptionsPanel(self)
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

    def Mesh_Options(self):
        print("Mesh")
        return QtW.QLabel("Vista Mesh", alignment=Qt.AlignCenter)

    def MField_Options(self):
        print("MField")
        return QtW.QLabel("Vista Campo Magnético", alignment=Qt.AlignCenter)

    def Density_Options(self):
        print("Density")
        return QtW.QLabel("Vista Densidad", alignment=Qt.AlignCenter)

    def Simulation_Options(self):
        print("Simulation")
        return QtW.QLabel("Vista Simulación", alignment=Qt.AlignCenter)

    @property
    def current_viewer(self):
        return self.View_Part.view_stack.currentWidget()

    def on_dynamic_tab_clicked(self, index, btn):
        self.parameters_view.setCurrentIndex(index)
        print("Index actual:", self.parameters_view.currentIndex())
        self.set_active_button(btn)
        if index == 0:
            self.View_Part.switch_view("mesh")
        elif index == 1:
            self.View_Part.switch_view("field")

    def set_active_button(self, active_btn):
        for btn in self.Options.tab_buttons:
            if btn == active_btn:
                btn.setStyleSheet(button_activate_style())
            else:
                btn.setStyleSheet(button_options_style())



