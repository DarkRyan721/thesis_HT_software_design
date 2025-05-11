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
from styles.stylesheets import *

class MainWindow(QtW.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HET simulator")
        self.setMinimumSize(1500, 800)
        self.resize(1500, 800)

        self.bool_hide_options = False
        self._setup_ui()
        self.frame.addWidget(self.Options, stretch=0.3)
        self.frame.addWidget(self.Parameters, stretch=1)
        self.frame.addWidget(self.View_Part, stretch=2)
        self.prev_params_mesh = (None, None, None)
        self.prev_params_electric_field = (None, None)
        self.setStyleSheet(self.self_Style())


        self.viewer = None
        self.tab_buttons = []
        self.parameters_views = None
        self.mesh_instance = None
        self.current_mesh = None
        self.prev_params_mesh = (None, None, None)
        self.prev_params_electric_field = (None, None)
        self.chk_show_axes = None
        self.chk_show_bounds = None
        self.combo_render_mode = None
        self.opacity_slider = None

    def _setup_ui(self):
        #_____________________________________________________________________________________________
        #                   Ventana principal

        central_widget = QtW.QWidget() #Frame base de aplicacion Qt
        self.frame = QtW.QHBoxLayout()  #Frame principal. Contiene las tres columnas visibles
        self.frame.setSpacing(0) # Espaciado cero entre
        self.frame.setContentsMargins(0,0,0,0) # Las columnas no tienen margen

        #_____________________________________________________________________________________________
        #                   Creacion de las tres columnas principales

        self.Parameters = self.set_Parameters()
        self.Options = self.set_Options()
        self.View_Part = self.set_View_Part()      


        self.frame.addWidget(self.Options, stretch=0.3)
        self.frame.addWidget(self.Parameters, stretch=1)
        self.frame.addWidget(self.View_Part, stretch=2)

        #_____________________________________________________________________________________________
        #                   Añadiendo las configuraciones al frame base y frame principal

        central_widget.setLayout(self.frame)
        self.setCentralWidget(central_widget)

    def set_Options(self):
        self.style_btn = button_style()
        self.style_btn_active = button_activate_style()
        #_____________________________________________________________________________________________
        #                   Creacion del frame base y layout vertical para [Options]

        frame_options = QtW.QFrame()
        frame_options.setStyleSheet("background-color: #131313; border-radius: 0px;")
        options_layout = QtW.QVBoxLayout()
        options_layout.setContentsMargins(0,0,0,0)

        #_____________________________________________________________________________________________
        #                   Creacion de componentes

        self.tab_buttons = []
        btn_info = [
            ("Home", "🏠"),
            ("Mesh", "▩"),
            ("LaPlace", "E"),
            ("Magnet", "B"),
            ("Density", "σ"),
            ("Simulation", "⯈")
        ]

        for index, (name, label) in enumerate(btn_info):
            btn = QtW.QPushButton(label)
            btn.setObjectName(name + "_Btn")
            if name == "Home" and label == "🏠":
                btn.setStyleSheet(self.style_btn_active)
            else:
                btn.setStyleSheet(self.style_btn)
            btn.setSizePolicy(QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Preferred)
            btn.setMinimumSize(60, 10)

            btn.clicked.connect(lambda _, i=index, b=btn: self.on_dynamic_tab_clicked(i, b))
            options_layout.addWidget(btn)
            self.tab_buttons.append(btn)
        options_layout.addStretch()
        self.parameters_views.addWidget(self.Home_Options())        # index 0
        self.parameters_views.addWidget(self.Mesh_Options())        # index 1
        self.parameters_views.addWidget(self.LaPlace_Options())     # index 2
        self.parameters_views.addWidget(self.MField_Options())      # index 3
        self.parameters_views.addWidget(self.Density_Options())     # index 4
        self.parameters_views.addWidget(self.Simulation_Options())  # index 5
        frame_options.setLayout(options_layout)
        return frame_options

    def set_Parameters(self):
        #_____________________________________________________________________________________________
        #                   Creacion del frame base y layout vertical para [Options]

        self.frame_parameters = QtW.QFrame()
        self.frame_parameters.setStyleSheet("background-color: #fbfcfc; border-radius: 0px;")
        self.frame_parameters.setFixedWidth(640)
        self.frame_parameters.setSizePolicy(QtW.QSizePolicy.Policy.Fixed, QtW.QSizePolicy.Policy.Expanding)
        parameters_layout = QtW.QVBoxLayout()

        self.parameters_views = QtW.QStackedWidget()
        self.parameters_views.setObjectName("Parameters_Views")
        parameters_layout.setContentsMargins(0, 5, 5, 5)
        self.parameters_views.setStyleSheet("""
        QStackedWidget#Parameters_Views {
            background-color: #2c2c2c;
            border-top-right-radius: 10px;
            border-bottom-right-radius: 10px;
        }
        """)

        parameters_layout.addWidget(self.parameters_views)
        self.frame_parameters.setLayout(parameters_layout)

        return self.frame_parameters

    def set_View_Part(self):
        frame_VPart = QtW.QFrame()
        frame_VPart.setObjectName("VPartFrame")

        VPart_layout = QtW.QVBoxLayout()
        VPart_layout.setContentsMargins(15, 15, 15, 15)

        self.viewer = QtInteractor(frame_VPart)
        self.viewer.setStyleSheet("background-color: #131313; border-radius: 5px;")
        self.viewer.set_background("gray")
        self.viewer.add_axes(interactive=True)
        self.viewer.setSizePolicy(QtW.QSizePolicy.Policy.Expanding, QtW.QSizePolicy.Policy.Expanding)


        # Paso 2: Crear worker y conectar señal
        self.worker = MeshLoaderWorker()
        self.worker.finished.connect(self.update_viewer)
        self.worker.start()
        self.viewer.view_zx()
        self.viewer.reset_camera()

        VPart_layout.addWidget(self.viewer)
        frame_VPart.setLayout(VPart_layout)

        return frame_VPart

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'frame_content'):
            new_width = int(self.width())
            new_height = int(self.height())
            self.frame_content.setMinimumSize(new_width, new_height)

    def on_dynamic_tab_clicked(self, index, btn):
        self.parameters_views.setCurrentIndex(index)
        self.set_active_button(btn)
        self.viewer.clear()
        self.viewer.reset_camera()
        if index == 2:  # LaPlace tab
            try:
                print("Cargando campo electrico")
                self.add_field_vectors_from_npy("./data_files/Electric_Field_np.npy")
            except Exception as e:
                print(f"⚠️ No se pudo cargar el campo eléctrico: {e}")

    def set_active_button(self, active_btn):
        for btn in self.tab_buttons:
            if btn == active_btn:
                btn.setStyleSheet(self.style_btn_active)
            else:
                btn.setStyleSheet(self.style_btn)

    def Home_Options(self):
        print("Home")
        # Widget principal
        home_widget = QtW.QWidget()
        layout = QtW.QVBoxLayout(home_widget)  # <== asocia el layout directamente al widget
        home_widget.setStyleSheet("background-color: transparent;")

        def input_with_unit(default_value, unit):
            field = QLineEdit(default_value)
            unit_lbl = QLabel(unit)
            unit_lbl.setStyleSheet("color: gray; padding-left: 5px;")

            box = QHBoxLayout()
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(field)
            box.addWidget(unit_lbl)

            container = QWidget()
            container.setLayout(box)
            return container, field

        # ----------------------------
        # Simulation Domain
        sim_box = QGroupBox("Simulation domain")
        sim_box.setSizePolicy(QtW.QSizePolicy.Expanding, QtW.QSizePolicy.Preferred)
        self.sim_layout = QFormLayout()
        # Simulation Domain
        self.input_L_container, self.input_L = input_with_unit("0.02", "[m]")
        self.input_R_Big_container, self.input_R_Big = input_with_unit("0.050", "[m]")
        self.input_R_Small_container, self.input_R_Small = input_with_unit("0.027", "[m]")

        self.sim_layout = QFormLayout()
        self.sim_layout.addRow("Total length (L):", self.input_L_container)
        self.sim_layout.addRow("Total radius (R):", self.input_R_Big_container)
        self.sim_layout.addRow("Total radius (R):", self.input_R_Small_container)
        sim_box.setLayout(self.sim_layout)
        layout.addWidget(sim_box)

        # ----------------------------
        update_btn = QPushButton("Update")
        update_btn.setStyleSheet("""
            QPushButton {
                background-color: #6082B6;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 6px;
                margin-top: 15px;
            }
            QPushButton:hover {
                background-color: #5c6bc0;
            }
            QPushButton:pressed {
                background-color: #303f9f;
            }
            QPushButton:disabled {
                background-color: #9e9e9e;
                color: #eeeeee;
            }
        """)

        update_btn.clicked.connect(self.on_update_clicked_mesh)
        layout.addWidget(update_btn)

        style_panel = self.create_style_panel()
        layout.addWidget(style_panel)
        layout.addStretch()
        return home_widget

    def Mesh_Options(self):
        print("Mesh")
        return QtW.QLabel("Vista Mesh", alignment=Qt.AlignCenter)

    def LaPlace_Options(self):
        print("LaPlace")

        EField_widget = QtW.QWidget()
        layout = QtW.QVBoxLayout(EField_widget)  # <== asocia el layout directamente al widget
        EField_widget.setStyleSheet("background-color: transparent;")

        def input_with_unit(default_value, unit):
            field = QLineEdit(default_value)
            unit_lbl = QLabel(unit)
            unit_lbl.setStyleSheet("color: gray; padding-left: 5px;")

            box = QHBoxLayout()
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(field)
            box.addWidget(unit_lbl)

            container = QWidget()
            container.setLayout(box)
            return container, field

        # ----------------------------
        # Simulation Domain
        field_box = QGroupBox("Electric Field Parameters")
        field_box.setSizePolicy(QtW.QSizePolicy.Expanding, QtW.QSizePolicy.Preferred)
        self.sim_layout = QFormLayout()
        # Simulation Domain
        self.input_Volt_container, self.Volt = input_with_unit("300", "[V]")
        self.input_Volt_Cath_container, self.Volt_Cath = input_with_unit("16", "[V]")

        self.sim_layout = QFormLayout()
        self.sim_layout.addRow("Voltaje (Volt):", self.input_Volt_container)
        self.sim_layout.addRow("Voltaje Cathode (Volt_Cath): ", self.input_Volt_Cath_container)
        field_box.setLayout(self.sim_layout)
        layout.addWidget(field_box)

        # ----------------------------
        update_btn = QPushButton("Update")
        update_btn.setStyleSheet(self.button_style())

        update_btn.clicked.connect(self.on_update_clicked_Electric_field)
        layout.addWidget(update_btn)


        style_panel = self.create_style_panel()
        layout.addWidget(style_panel)
        layout.addStretch()


        return EField_widget

    def add_field_vectors_from_npy(self, npy_path, scale=0.01):
        data = np.load(npy_path)
        if data.shape[1] < 6:
            raise ValueError("El archivo .npy debe contener [x, y, z, Ex, Ey, Ez]")

        points = data[:, :3]
        vectors = data[:, 3:]
        magnitudes = np.linalg.norm(vectors, axis=1)
        log_magnitudes = np.log10(magnitudes + 1e-3)

        mesh = pv.PolyData(points)
        mesh["vectors"] = vectors
        mesh["magnitude"] = log_magnitudes

        self.viewer.clear()
        self.viewer.add_mesh(mesh.glyph(orient="vectors", scale=False, factor=scale),
                            scalars="magnitude", cmap="plasma")
        self.viewer.add_axes(interactive=True)
        self.viewer.reset_camera()

    def MField_Options(self):
        print("MField")
        return QtW.QLabel("Vista Campo Magnético", alignment=Qt.AlignCenter)

    def Density_Options(self):
        print("Density")
        return QtW.QLabel("Vista Densidad", alignment=Qt.AlignCenter)

    def Simulation_Options(self):
        print("Simulation")
        return QtW.QLabel("Vista Simulación", alignment=Qt.AlignCenter)

    def Hide_Options(self):
        self.bool_hide_options = not self.bool_hide_options

        if self.bool_hide_options == True:
            self.frame.setStretch(1,0.8)
            self.Hide_Btn.setText("≡")
        else:
            #self.centralWidget().layout().setStretch(0,1)
            self.frame.setStretch(1,1)
            self.Hide_Btn.setText("≡ Ocultar")

    def on_update_clicked_mesh(self):

        H = float(self.input_L.text())
        R_Big = float(self.input_R_Big.text())
        R_Small = float(self.input_R_Small.text())

        print(f"L = {H}, R = {R_Big}, R = {R_Small}")
        print(self.prev_params_mesh)
        new_params = (H, R_Big, R_Small)
        if new_params != self.prev_params_mesh:
            print("🔄 ¡Parámetros cambiaron:", new_params)
            self.prev_params_mesh = new_params
            self.mesh_instance = HallThrusterMesh(R_big=R_Big, R_small=R_Small, H=H)
            self.mesh_instance.generate()

            # ✅ Solo lectura y visualización en background
            self.worker = MeshLoaderWorker()
            self.worker.finished.connect(self.update_viewer)
            self.worker.start()
        else:
            print("⚠️ No se han realizado cambios en la malla.")

    def on_update_clicked_Electric_field(self):
        voltaje = float(self.Volt.text())
        voltaje_Cath = float(self.Volt_Cath.text())

        print(f"voltaje = {voltaje}, voltaje_cath = {voltaje_Cath}")
        print(self.prev_params_electric_field)
        new_params = (voltaje, voltaje_Cath)
        if new_params != self.prev_params_electric_field:
            print("🔄 ¡Parámetros cambiaron:", new_params)
            self.prev_params_electric_field = new_params
            solver_electric_field = ElectricFieldSolver()

        else:
            print("⚠️ No se han realizado cambios en la malla.")

    def update_viewer(self, mesh):
        self.current_mesh = mesh

        def deferred_render():
            self.apply_visual_style()

        QtCore.QTimer.singleShot(0, deferred_render)

    def create_style_panel(self):
        panel = QtW.QGroupBox("Visual Style")
        layout = QtW.QVBoxLayout(panel)

        self.combo_render_mode = QtW.QComboBox()
        self.combo_render_mode.addItems(["Surface", "Wireframe", "Points", "Surface with edges"])
        self.combo_render_mode.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: white;
                selection-background-color: #5c6bc0;
                border: none;
            }
        """)
        layout.addWidget(QtW.QLabel("Render Mode"))
        layout.addWidget(self.combo_render_mode)

        # Crear contenedor horizontal
        checkbox_container = QtW.QWidget()
        checkbox_layout = QtW.QHBoxLayout()
        checkbox_layout.setContentsMargins(0, 0, 0, 0)

        self.chk_show_axes = QtW.QCheckBox("Show Axes")
        self.chk_show_bounds = QtW.QCheckBox("Show Bounds")
        self.chk_show_axes.setStyleSheet(self.checkbox_style())
        self.chk_show_bounds.setStyleSheet(self.checkbox_style())

        checkbox_layout.addWidget(self.chk_show_axes)
        checkbox_layout.addWidget(self.chk_show_bounds)
        checkbox_layout.addStretch()

        checkbox_container.setLayout(checkbox_layout)
        layout.addWidget(checkbox_container)


        self.opacity_slider = QtW.QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        layout.addWidget(QtW.QLabel("Opacity"))
        layout.addWidget(self.opacity_slider)

        apply_btn = QtW.QPushButton("Apply Style")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #6082B6;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 6px;
                margin-top: 15px;
            }
            QPushButton:hover {
                background-color: #5c6bc0;
            }
            QPushButton:pressed {
                background-color: #303f9f;
            }
            QPushButton:disabled {
                background-color: #9e9e9e;
                color: #eeeeee;
            }
        """)
        apply_btn.clicked.connect(self.apply_visual_style)
        layout.addWidget(apply_btn)
        layout.addStretch()

        return panel

    def apply_visual_style(self):
        if not hasattr(self, 'current_mesh'):
            print("⚠️ No hay malla cargada.")
            return

        self.viewer.clear()
        mode    = self.combo_render_mode.currentText()
        opacity = self.opacity_slider.value() / 100.0

        # Parámetros comunes
        common_kwargs = dict(
            color="#dddddd",
            opacity=opacity,
            lighting=True,
            smooth_shading=True,
            ambient=0.3,
            diffuse=1.0,
            specular=0.4,
        )

        #  ▷ Por defecto no pasamos ni 'style' ni 'show_edges'
        if mode == "Surface":
            # sólo caras rellenas
            self.viewer.add_mesh(self.current_mesh,color="white", show_edges=False,
                            opacity=1.0, lighting=True, smooth_shading=True,
                            ambient=0.3, diffuse=1.0, specular=0.4
                            )

        elif mode == "Surface with edges":
            # caras rellenas + aristas
            self.viewer.add_mesh(
                self.current_mesh,
                style="surface",
                show_edges=True,
                edge_color="black",
                color="#dddddd",
                opacity=opacity,
                lighting=True,
                smooth_shading=True,
                ambient=0.3,
                diffuse=1.0,
                specular=0.4
            )

        elif mode == "Wireframe":
            self.viewer.add_mesh(
                self.current_mesh,
                style="wireframe",
                **common_kwargs
            )

        elif mode == "Points":
            self.viewer.add_mesh(
                self.current_mesh,
                style="points",
                **common_kwargs
            )

        # Ejes y límites, según los checkboxes
        if self.chk_show_axes.isChecked():
            self.viewer.add_axes(interactive=True)
        if self.chk_show_bounds.isChecked():
            self.viewer.show_bounds(
                all_edges=True,
                color="white",
                location="outer",
                grid=True
            )
        self.viewer.reset_camera()


    def add_potential_field_from_npy(viewer, npy_path):
        """
        Carga y visualiza el potencial eléctrico escalar desde un archivo .npy.

        Parámetros:
        - viewer: Instancia de QtInteractor
        - npy_path: Ruta al archivo con [x, y, z, phi]
        """
        import numpy as np
        import pyvista as pv

        data = np.load(npy_path)
        if data.shape[1] < 4:
            raise ValueError("El archivo .npy debe contener [x, y, z, phi]")

        points = data[:, :3]
        phi = data[:, 3]

        mesh = pv.PolyData(points)
        mesh["potential"] = phi

        viewer.clear()
        viewer.add_mesh(mesh, scalars="potential", cmap="viridis", point_size=5)
        viewer.add_axes(interactive=True)
        viewer.reset_camera()


class MeshLoaderWorker(QThread):
    finished = Signal(object)

    def __init__(self):
        super().__init__()
        self.msh = meshio.read("./data_files/SimulationZone.msh")

    def run(self):
        cells = self.msh.cells_dict.get("triangle")
        if cells is None:
            return

        num_cells = cells.shape[0]
        faces = np.hstack([np.full((num_cells, 1), 3), cells]).astype(np.int32).flatten()
        mesh = pv.PolyData(self.msh.points, faces)

        self.finished.emit(mesh)
