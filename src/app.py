import sys
from PySide6.QtWidgets import QApplication
from window import MainWindow  # Asegúrate que este nombre coincide con tu archivo .py
import os
import PySide6

def run_app():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
