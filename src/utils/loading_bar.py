from PySide6.QtWidgets import QWidget, QVBoxLayout, QProgressBar, QLabel
from PySide6.QtCore import Qt

class ProgressBarWidget(QWidget):
    def __init__(self, text="Procesando..."):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(text)
        self.label.setAlignment(Qt.AlignCenter)
        self.progress = QProgressBar()
        # Barra en modo indeterminado
        self.progress.setRange(0, 0)   # ¡Este es el cambio clave!
        self.progress.setTextVisible(False)
        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        self.setLayout(layout)
        self.hide()  # Empieza oculta

    def start(self, text=None):
        if text:
            self.label.setText(text)
        self.show()

    def finish(self):
        self.hide()
