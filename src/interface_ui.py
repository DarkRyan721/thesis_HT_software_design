import sys
from PyQt5.QtWidgets import QApplication, QWidget

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle('Mi primera ventana PyQt')
window.show()
sys.exit(app.exec_())
