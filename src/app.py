import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow  # Asegúrate que este nombre coincide con tu archivo .py
import os
import PySide6
import signal
import sys
from project_paths import *
def run_app():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Permite Ctrl+C cerrar la app
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
