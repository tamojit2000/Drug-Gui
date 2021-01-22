from PyQt5 import QtWidgets, uic, QtGui, QtCore
import sys
import cv2
from functions import *

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('basic.ui', self)
        self.PATH=None
        self.pushButton.clicked.connect(self.browse)
        self.pushButton_2.clicked.connect(self.Predict)
        self.pushButton_3.clicked.connect(self.About)
        self.show()

    def browse(self):
        fm = QtWidgets.QFileDialog.getOpenFileName(None,'Browse File')
        filename = fm[0]
        if filename=='': filename='a.jpg'                
        self.label.setPixmap(QtGui.QPixmap(filename))
        self.PATH=filename
        print('File Path:',self.PATH)
        self.label_4.setText('Result')

    def Predict(self):
        print('predict')
        print('File Path:',self.PATH)
        prediction=get_prediction(self.PATH)
        print('Prediction:',prediction)
        self.label_4.setText(prediction)
        

    def About(self):
        print('About')
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("About")
        msg.setText(ABOUT_TEXT)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.exec_()

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
