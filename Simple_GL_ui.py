# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Simple_GL_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(991, 560)
        self.groupBox_3 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 10, 961, 531))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setMouseTracking(False)
        self.groupBox_3.setCheckable(False)
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalSlider_zoom = QtWidgets.QSlider(self.groupBox_3)
        self.horizontalSlider_zoom.setGeometry(QtCore.QRect(130, 480, 181, 22))
        self.horizontalSlider_zoom.setMaximum(200)
        self.horizontalSlider_zoom.setProperty("value", 150)
        self.horizontalSlider_zoom.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_zoom.setObjectName("horizontalSlider_zoom")
        self.label_20 = QtWidgets.QLabel(self.groupBox_3)
        self.label_20.setGeometry(QtCore.QRect(80, 480, 41, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.pushButton_Exit = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_Exit.setGeometry(QtCore.QRect(790, 470, 112, 34))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_Exit.setFont(font)
        self.pushButton_Exit.setObjectName("pushButton_Exit")
        self.openGLWidget = QtWidgets.QOpenGLWidget(self.groupBox_3)
        self.openGLWidget.setGeometry(QtCore.QRect(10, 30, 941, 431))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.openGLWidget.sizePolicy().hasHeightForWidth())
        self.openGLWidget.setSizePolicy(sizePolicy)
        self.openGLWidget.setMouseTracking(False)
        self.openGLWidget.setObjectName("openGLWidget")
        self.pushButton_start = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_start.setGeometry(QtCore.QRect(420, 470, 112, 34))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_start.setFont(font)
        self.pushButton_start.setObjectName("pushButton_start")
        self.pushButton_switch = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_switch.setGeometry(QtCore.QRect(600, 480, 121, 41))
        self.pushButton_switch.setObjectName("pushButton_switch")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "OpenGL"))
        self.groupBox_3.setTitle(_translate("Dialog", "Jolly Green Giant"))
        self.label_20.setText(_translate("Dialog", "Zoom"))
        self.pushButton_Exit.setText(_translate("Dialog", "Exit"))
        self.pushButton_start.setText(_translate("Dialog", "Start"))
        self.pushButton_switch.setText(_translate("Dialog", "switch"))

