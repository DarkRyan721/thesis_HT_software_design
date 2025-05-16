import sys
import os
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

# Añadir ../../ (es decir, src/) al path para importar desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QPushButton
)
from PySide6.QtCore import Qt
import PySide6.QtWidgets as QtW

from utils.ui_helpers import _input_with_unit
from styles.stylesheets import button_parameters_style
# TODO: Importar el solver y el loader apropiados para el campo magnético
# from E_field_solver import MagneticFieldSolver
# from utils.magnetic_loader import MagneticLoaderWorker

class MagneticOptionsPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.simulation_state = self.main_window.simulation_state
        self.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(self)

        # Parámetros magnéticos
        self.input_nSteps_container, self.input_nSteps = _input_with_unit("1500", "[N]")
        self.input_L_container, self.input_L = _input_with_unit("0.021", "[m]")
        self.input_Rin_container, self.input_Rin = _input_with_unit("0.027", "[m]")
        self.input_Rout_container, self.input_Rout = _input_with_unit(str(0.8*0.027), "[m]")
        self.input_Rext_container, self.input_Rext = _input_with_unit("0.05", "[m]")
        self.input_N_container, self.input_N = _input_with_unit("150", "[N]")
        self.input_i_container, self.input_i = _input_with_unit("15", "[A]")

        # Visualizador 3D
        self.magnetic_viewer = self._create_viewer()

        mag_box = QGroupBox("Magnetic Field Parameters")
        mag_box.setSizePolicy(QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Fixed)
        form = QFormLayout()
        form.addRow("nSteps:", self.input_nSteps_container)
        form.addRow("Longitud L:", self.input_L_container)
        form.addRow("Radio interior Rin:", self.input_Rin_container)
        form.addRow("Radio exterior Rout:", self.input_Rout_container)
        form.addRow("Radio de extensión Rext:", self.input_Rext_container)
        form.addRow("Número de vueltas N:", self.input_N_container)
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
        # Leer valores de los inputs
        nSteps = int(self.input_nSteps.text())
        L = float(self.input_L.text())
        Rin = float(self.input_Rin.text())
        Rout = float(self.input_Rout.text())
        Rext = float(self.input_Rext.text())
        N = int(self.input_N.text())
        muo = float(self.input_muo.text())
        i = float(self.input_i.text())

        # Actualizar el estado de la simulación
        self.simulation_state.nSteps = nSteps
        self.simulation_state.L = L
        self.simulation_state.Rin = Rin
        self.simulation_state.Rout = Rout
        self.simulation_state.Rext = Rext
        self.simulation_state.N = N
        self.simulation_state.muo = muo
        self.simulation_state.i = i

        new_params = (nSteps, L, Rin, Rout, Rext, N, muo, i)
        if new_params != getattr(self.simulation_state, 'prev_params_magnetic', None):
            print("🔄 Parámetros magnéticos cambiaron:", new_params)
            self.simulation_state.prev_params_magnetic = new_params
            # TODO: Llamar al solver y guardar resultado
            # solver = MagneticFieldSolver()
            # field_data = solver.compute(nSteps, L, Rin, Rout, Rext, N, muo, i)
            # npy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data_files/Magnetic_Field.npy"))
            # solver.save(field_data, filename=npy_path)
            # Cargar y visualizar con loader
            # self.worker = MagneticLoaderWorker(mode="magnetic")
            # self.worker.finished.connect(self.on_magnetic_loaded)
            # self.worker.start()
        else:
            print("⚠️ No se han realizado cambios en los parámetros magnéticos.")

    def on_magnetic_loaded(self, data):
        self.current_magnetic = data
        # Cambiar vista si está activo el panel de magnético (asumiendo índice 3)
        if self.main_window.parameters_view.currentIndex() == 3:
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
        path = os.path.join(os.path.dirname(__file__), "../../data_files/Magnetic_Field.npy")
        if os.path.exists(path):
            print("🔗 Cargando campo magnético inicial...")
            # self.worker = MagneticLoaderWorker(mode="magnetic")
            # self.worker.finished.connect(self.on_magnetic_loaded)
            # self.worker.start()
