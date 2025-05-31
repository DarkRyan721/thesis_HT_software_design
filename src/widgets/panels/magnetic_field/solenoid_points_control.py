from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QCheckBox, QPushButton, QHBoxLayout

from gui_styles.stylesheets import button_parameters_style

class SolenoidPointsControlPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Opciones de Visualización de Solenoides", parent)
        layout = QVBoxLayout(self)

        # Checkboxes para cada solenoide
        self.cb_inner = QCheckBox("Solenoide Interno")
        self.cb_inner.setChecked(True)
        self.cb_1 = QCheckBox("Solenoide 1")
        self.cb_1.setChecked(True)
        self.cb_2 = QCheckBox("Solenoide 2")
        self.cb_2.setChecked(True)
        self.cb_3 = QCheckBox("Solenoide 3")
        self.cb_3.setChecked(True)
        self.cb_4 = QCheckBox("Solenoide 4")
        self.cb_4.setChecked(True)
        self.cb_cylinder = QCheckBox("Cilindro Externo")
        self.cb_cylinder.setChecked(True)

        # Crear las filas de checkboxes
        row1 = QHBoxLayout()
        row1.addWidget(self.cb_inner)
        row1.addWidget(self.cb_1)
        row1.addWidget(self.cb_2)

        row2 = QHBoxLayout()
        row2.addWidget(self.cb_3)
        row2.addWidget(self.cb_4)
        row2.addWidget(self.cb_cylinder)

        # Añadir filas al layout principal
        layout.addLayout(row1)
        layout.addLayout(row2)

        # Botón para actualizar gráfico
        self.btn_update = QPushButton("Actualizar Gráfico")
        self.btn_update.setStyleSheet(button_parameters_style())
        layout.addWidget(self.btn_update)

        self.setLayout(layout)

    def get_params(self):
        return dict(
            Solenoid_inner=self.cb_inner.isChecked(),
            Solenoid_1=self.cb_1.isChecked(),
            Solenoid_2=self.cb_2.isChecked(),
            Solenoid_3=self.cb_3.isChecked(),
            Solenoid_4=self.cb_4.isChecked(),
            Cylinder_ext=self.cb_cylinder.isChecked()
        )