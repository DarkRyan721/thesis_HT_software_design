from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QGroupBox,
    QLabel, QLineEdit, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QLabel,
    QHBoxLayout, QPushButton, QCheckBox, QSlider, QComboBox
)
from PySide6.QtWidgets import QStyle
class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: none;
                font-weight: bold;
                text-align: left;
                padding: 8px;
            }
            QPushButton:checked {
                background-color: #e0e0e0;
            }
        """)

        self.toggle_button.setIcon(self.style().standardIcon(getattr(QStyle, 'SP_ArrowRight')))
        self.toggle_button.toggled.connect(self.toggle_content)

        self.toggle_button.setStyleSheet("""
            QGroupBox {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                  stop: 0 #E0E0E0, stop: 1 #FFFFFF);
                border: 2px solid gray;
                border-radius: 5px;
                margin-top: 1ex; /* Deja espacio en la parte superior para el título */
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* Posiciona el título en la parte superior central */
                padding: 0 3px;
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                  stop: 0 #FF0ECE, stop: 1 #FFFFFF);
            }
        """)

        self.content_area = QFrame()
        self.content_area.setFrameShape(QFrame.StyledPanel)
        self.content_area.setVisible(False)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.toggle_button)
        self.layout.addWidget(self.content_area)

    def setContentLayout(self, content_layout):
        self.content_area.setLayout(content_layout)

    def toggle_content(self, checked):
        self.content_area.setVisible(checked)
        # Cambia el ícono según el estado
        icon = self.style().standardIcon(
            getattr(QStyle, 'SP_ArrowDown') if checked else getattr(QStyle, 'SP_ArrowRight')
        )
        self.toggle_button.setIcon(icon)
