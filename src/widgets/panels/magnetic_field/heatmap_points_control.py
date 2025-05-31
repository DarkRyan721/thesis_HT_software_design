from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QCheckBox, QPushButton, QHBoxLayout, QLabel, QComboBox, QDoubleSpinBox

from gui_styles.stylesheets import button_parameters_style

class HeatmapControlPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Opciones de Heatmap", parent)
        layout = QVBoxLayout(self)

        # Plano: XY o ZX
        self.cb_xy = QCheckBox("Plano XY")
        self.cb_xy.setChecked(True)
        self.cb_zx = QCheckBox("Plano ZX")

        row1 = QHBoxLayout()
        row1.addWidget(self.cb_xy)
        row1.addWidget(self.cb_zx)

        # Solenoide central, todos
        self.cb_center = QCheckBox("Solenoide Central")
        self.cb_center.setChecked(True)
        self.cb_all = QCheckBox("Todos los Solenoides")
        self.cb_all.setChecked(True)

        row2 = QHBoxLayout()
        row2.addWidget(self.cb_center)
        row2.addWidget(self.cb_all)

        # Valor del plano (Z o Y según plano)
        self.label_plane_value = QLabel("Valor plano (Z/Y):")
        self.spin_plane_value = QDoubleSpinBox()
        self.spin_plane_value.setRange(-1.0, 1.0)
        self.spin_plane_value.setDecimals(3)
        self.spin_plane_value.setSingleStep(0.01)
        self.spin_plane_value.setValue(0.01)

        row3 = QHBoxLayout()
        row3.addWidget(self.label_plane_value)
        row3.addWidget(self.spin_plane_value)

        # Resolución
        self.label_resolution = QLabel("Resolución:")
        self.spin_resolution = QDoubleSpinBox()
        self.spin_resolution.setRange(10, 500)
        self.spin_resolution.setValue(100)
        self.spin_resolution.setSingleStep(10)

        row4 = QHBoxLayout()
        row4.addWidget(self.label_resolution)
        row4.addWidget(self.spin_resolution)

        # Botón actualizar
        self.btn_update = QPushButton("Actualizar Heatmap")
        self.btn_update.setStyleSheet(button_parameters_style())

        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)
        layout.addLayout(row4)
        layout.addWidget(self.btn_update)
        self.setLayout(layout)

    def get_params(self):
        return dict(
            XY=self.cb_xy.isChecked(),
            ZX=self.cb_zx.isChecked(),
            Solenoid_Center=self.cb_center.isChecked(),
            All_Solenoids=self.cb_all.isChecked(),
            Plane_Value=self.spin_plane_value.value(),
            resolution=int(self.spin_resolution.value())
        )
