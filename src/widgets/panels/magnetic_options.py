import sys
import os
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from magnetic_field_noGPU import B_Field

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
        self.input_nSteps_container, self.input_nSteps = _input_with_unit(str(self.simulation_state.nSteps), "[N]")
        self.input_N_container, self.input_N = _input_with_unit(str(self.simulation_state.N), "[N]")
        self.input_i_container, self.input_i = _input_with_unit(str(self.simulation_state.I), "[A]")

        # Visualizador 3D
        self.magnetic_viewer = self._create_viewer()

        mag_box = QGroupBox("Magnetic Field Parameters")
        mag_box.setSizePolicy(QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Fixed)
        form = QFormLayout()
        form.addRow("nSteps:", self.input_nSteps_container)
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
        N = int(self.input_N.text())
        I = float(self.input_i.text())

        # Actualizar el estado de la simulación
        self.simulation_state.nSteps = nSteps
        self.simulation_state.N = N
        self.simulation_state.I = I

        new_params = (nSteps, N, I)
        if new_params != self.simulation_state.prev_params_magnetic:
            print("🔄 Parámetros magnéticos cambiaron:", new_params)
            self.simulation_state.prev_params_magnetic = new_params
            magnetic_instance = B_Field(nSteps=self.simulation_state.nSteps, N = self.simulation_state.N, I=self.simulation_state.I)

            E_File = np.load("data_files/Electric_Field_np.npy")
            spatial_coords = E_File[:, :3]

            # ⚠️ Aquí eliges si quieres solo el centro o todos
            B_value = magnetic_instance.Total_Magnetic_Field(S=spatial_coords)


            points = spatial_coords
            vectors = B_value
            magnitudes = np.linalg.norm(vectors, axis=1)

            # Crea el objeto PyVista
            B_field_pv = pv.PolyData(points)
            B_field_pv["vectors"] = vectors
            B_field_pv["magnitude"] = magnitudes

            self.magnetic_viewer.clear()
            self.magnetic_viewer.add_mesh(B_field_pv, scalars="magnitude", cmap="viridis")
            self.magnetic_viewer.add_arrows(B_field_pv.points, vectors, mag=0.01, label="B-field")  # Opcional
            self.magnetic_viewer.reset_camera()
            magnetic_instance.Save_B_Field(B=B_value, S=spatial_coords)

        else:
            print("⚠️ No se han realizado cambios en los parámetros magnéticos.")
        self.simulation_state.print_state()


    def on_magnetic_loaded(self, data):
        self.current_magnetic = data
        self.main_window.View_Part.current_data = data  # asegúrate de sincronizar ViewPanel
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
