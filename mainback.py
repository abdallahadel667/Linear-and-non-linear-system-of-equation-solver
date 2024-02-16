import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLineEdit, QLabel
from PyQt5.QtCore import QTimer, QUrl,QEventLoop
from PyQt5.QtMultimedia import QSound, QMediaPlayer, QMediaContent
from PyQt5 import QtGui, QtCore
from frontend import Ui_MainWindow
from iterative import Ui_Iterative
from lu import Ui_luwid
from gausswid import Ui_gausswidget
import copy
import math
from mpmath import mp, matrix
import time
from selection import Ui_MainWindow3
from frontend import Ui_MainWindow
from gui import Ui_MainWindow2
from backnon import Solver
from backend import Solver1



class Home(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow3()
        self.ui.setupUi(self)
        self.setWindowTitle("Solver")

        self.ui.pushButton.clicked.connect(self.first)
        self.ui.pushButton_2.clicked.connect(self.second)
    def first(self):
        self.linear= Solver1()
        self.linear.show()
        self.close()

    def second(self):
        self.non=Solver()
        self.non.show()
        self.close()






if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Home()
    window.show()
    sys.exit(app.exec_())
