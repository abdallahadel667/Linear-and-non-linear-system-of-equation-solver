# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gausswidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_gausswidget(object):
    def setupUi(self, gausswidget):
        gausswidget.setObjectName("gausswidget")
        gausswidget.resize(639, 467)
        self.layoutWidget = QtWidgets.QWidget(gausswidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 230, 351, 151))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.label_13 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(50)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_5.addWidget(self.label_13)
        self.solutionconstgrid = QtWidgets.QGridLayout()
        self.solutionconstgrid.setContentsMargins(-1, 42, -1, 25)
        self.solutionconstgrid.setVerticalSpacing(0)
        self.solutionconstgrid.setObjectName("solutionconstgrid")
        self.horizontalLayout_5.addLayout(self.solutionconstgrid)
        self.label_14 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(50)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_5.addWidget(self.label_14)
        self.label_17 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.horizontalLayout_5.addWidget(self.label_17)
        self.label_18 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(50)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.horizontalLayout_5.addWidget(self.label_18)
        self.solutiongrid = QtWidgets.QGridLayout()
        self.solutiongrid.setContentsMargins(-1, 20, -1, 0)
        self.solutiongrid.setVerticalSpacing(7)
        self.solutiongrid.setObjectName("solutiongrid")
        self.horizontalLayout_5.addLayout(self.solutiongrid)
        self.label_19 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(50)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.horizontalLayout_5.addWidget(self.label_19)
        self.layoutWidget_2 = QtWidgets.QWidget(gausswidget)
        self.layoutWidget_2.setGeometry(QtCore.QRect(10, 30, 621, 141))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget_2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(50)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.auggrid = QtWidgets.QGridLayout()
        self.auggrid.setObjectName("auggrid")
        self.horizontalLayout.addLayout(self.auggrid)
        self.label_7 = QtWidgets.QLabel(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(50)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.next = QtWidgets.QPushButton(self.layoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.next.sizePolicy().hasHeightForWidth())
        self.next.setSizePolicy(sizePolicy)
        self.next.setObjectName("next")
        self.horizontalLayout.addWidget(self.next)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)

        self.retranslateUi(gausswidget)
        QtCore.QMetaObject.connectSlotsByName(gausswidget)

    def retranslateUi(self, gausswidget):
        _translate = QtCore.QCoreApplication.translate
        gausswidget.setWindowTitle(_translate("gausswidget", "Form"))
        self.label_13.setText(_translate("gausswidget", "("))
        self.label_14.setText(_translate("gausswidget", ")"))
        self.label_17.setText(_translate("gausswidget", "="))
        self.label_18.setText(_translate("gausswidget", "("))
        self.label_19.setText(_translate("gausswidget", ")"))
        self.label.setText(_translate("gausswidget", "A*="))
        self.label_3.setText(_translate("gausswidget", "("))
        self.label_7.setText(_translate("gausswidget", ")"))
        self.next.setText(_translate("gausswidget", "Next"))
