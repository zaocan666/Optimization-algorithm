from PyQt5.QtWidgets import QTabWidget, QMainWindow, QDesktopWidget, QApplication
from PyQt5.QtGui import QIcon
import sys
from ui_TSP import Ui_TSP

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        height = 800
        self.resize(1200, height)
        self.center()
        self.setWindowTitle('TSP')
        # self.setWindowIcon(QIcon("tangram.ico"))

        self.tabW = QTabWidget(parent=self)
        ui_basic = Ui_TSP()

        self.tabW.addTab(ui_basic, "TSP问题")
        self.tabW.resize(1200,height)
        ui_basic.initUI()

        self.show()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)

if __name__ == "__main__":
    app = QApplication([])
    ui = MainWindow()
    sys.exit(app.exec_())