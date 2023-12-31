# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_VIO_alignment_Tool(object):
    def setupUi(self, VIO_alignment_Tool):
        VIO_alignment_Tool.setObjectName("VIO_alignment_Tool")
        VIO_alignment_Tool.resize(816, 577)
        VIO_alignment_Tool.setMinimumSize(QtCore.QSize(709, 577))
        self.centralwidget = QtWidgets.QWidget(VIO_alignment_Tool)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 2, 1, 1)
        self.range_spinbox = QtWidgets.QSpinBox(self.centralwidget)
        self.range_spinbox.setMaximum(10000)
        self.range_spinbox.setProperty("value", 1)
        self.range_spinbox.setObjectName("range_spinbox")
        self.gridLayout.addWidget(self.range_spinbox, 3, 3, 1, 1)
        self.Start_from = QtWidgets.QSlider(self.centralwidget)
        self.Start_from.setMaximum(100)
        self.Start_from.setPageStep(1)
        self.Start_from.setOrientation(QtCore.Qt.Horizontal)
        self.Start_from.setObjectName("Start_from")
        self.gridLayout.addWidget(self.Start_from, 9, 1, 1, 1)
        self.vio_shifter = QtWidgets.QSlider(self.centralwidget)
        self.vio_shifter.setMinimumSize(QtCore.QSize(200, 0))
        self.vio_shifter.setMaximumSize(QtCore.QSize(300, 16777215))
        self.vio_shifter.setPageStep(1)
        self.vio_shifter.setOrientation(QtCore.Qt.Horizontal)
        self.vio_shifter.setObjectName("vio_shifter")
        self.gridLayout.addWidget(self.vio_shifter, 5, 2, 1, 2)
        self.gt_shifter = QtWidgets.QSlider(self.centralwidget)
        self.gt_shifter.setMinimumSize(QtCore.QSize(200, 0))
        self.gt_shifter.setMaximumSize(QtCore.QSize(300, 16777215))
        self.gt_shifter.setPageStep(1)
        self.gt_shifter.setOrientation(QtCore.Qt.Horizontal)
        self.gt_shifter.setObjectName("gt_shifter")
        self.gridLayout.addWidget(self.gt_shifter, 7, 2, 1, 2)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 10, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 2, 1, 1)
        self.canvas = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.canvas.sizePolicy().hasHeightForWidth())
        self.canvas.setSizePolicy(sizePolicy)
        self.canvas.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.canvas.setFrameShadow(QtWidgets.QFrame.Raised)
        self.canvas.setObjectName("canvas")
        self.gridLayout.addWidget(self.canvas, 0, 0, 8, 2)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 9, 0, 1, 1)
        self.nav_offset_spinbox = QtWidgets.QSpinBox(self.centralwidget)
        self.nav_offset_spinbox.setProperty("value", 0)
        self.nav_offset_spinbox.setObjectName("nav_offset_spinbox")
        self.gridLayout.addWidget(self.nav_offset_spinbox, 4, 3, 1, 1)
        self.offset_spinbox = QtWidgets.QSpinBox(self.centralwidget)
        self.offset_spinbox.setMaximum(10000)
        self.offset_spinbox.setObjectName("offset_spinbox")
        self.gridLayout.addWidget(self.offset_spinbox, 6, 3, 1, 1)
        self.End_to = QtWidgets.QSlider(self.centralwidget)
        self.End_to.setMaximum(100)
        self.End_to.setPageStep(1)
        self.End_to.setOrientation(QtCore.Qt.Horizontal)
        self.End_to.setInvertedAppearance(True)
        self.End_to.setObjectName("End_to")
        self.gridLayout.addWidget(self.End_to, 10, 1, 1, 1)
        self.GT_file = QtWidgets.QPushButton(self.centralwidget)
        self.GT_file.setObjectName("GT_file")
        self.gridLayout.addWidget(self.GT_file, 0, 2, 1, 1)
        self.yscale_spinbox = QtWidgets.QSpinBox(self.centralwidget)
        self.yscale_spinbox.setMinimum(1)
        self.yscale_spinbox.setMaximum(10000)
        self.yscale_spinbox.setObjectName("yscale_spinbox")
        self.gridLayout.addWidget(self.yscale_spinbox, 1, 3, 1, 1)
        self.VIO_file = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.VIO_file.sizePolicy().hasHeightForWidth())
        self.VIO_file.setSizePolicy(sizePolicy)
        self.VIO_file.setObjectName("VIO_file")
        self.gridLayout.addWidget(self.VIO_file, 0, 3, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 6, 2, 1, 1)
        self.xscale_spinbox = QtWidgets.QSpinBox(self.centralwidget)
        self.xscale_spinbox.setMinimum(1)
        self.xscale_spinbox.setMaximum(10000)
        self.xscale_spinbox.setObjectName("xscale_spinbox")
        self.gridLayout.addWidget(self.xscale_spinbox, 2, 3, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 2, 1, 1)
        self.save_result = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.save_result.sizePolicy().hasHeightForWidth())
        self.save_result.setSizePolicy(sizePolicy)
        self.save_result.setObjectName("save_result")
        self.gridLayout.addWidget(self.save_result, 9, 3, 2, 1)
        VIO_alignment_Tool.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(VIO_alignment_Tool)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 816, 22))
        self.menubar.setObjectName("menubar")
        VIO_alignment_Tool.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(VIO_alignment_Tool)
        self.statusbar.setObjectName("statusbar")
        VIO_alignment_Tool.setStatusBar(self.statusbar)

        self.retranslateUi(VIO_alignment_Tool)
        QtCore.QMetaObject.connectSlotsByName(VIO_alignment_Tool)

    def retranslateUi(self, VIO_alignment_Tool):
        _translate = QtCore.QCoreApplication.translate
        VIO_alignment_Tool.setWindowTitle(_translate("VIO_alignment_Tool", "VIO_alignment_Tool"))
        self.label_2.setText(_translate("VIO_alignment_Tool", "Y_scale"))
        self.label_7.setText(_translate("VIO_alignment_Tool", "End To"))
        self.label_4.setText(_translate("VIO_alignment_Tool", "Show range"))
        self.label_3.setText(_translate("VIO_alignment_Tool", "Start From"))
        self.GT_file.setText(_translate("VIO_alignment_Tool", "Open GT file"))
        self.VIO_file.setText(_translate("VIO_alignment_Tool", "Open VIO path"))
        self.label_6.setText(_translate("VIO_alignment_Tool", "Shift GT data"))
        self.label.setText(_translate("VIO_alignment_Tool", "X scale"))
        self.label_5.setText(_translate("VIO_alignment_Tool", "Shift VIO data"))
        self.save_result.setText(_translate("VIO_alignment_Tool", "Save Result"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    VIO_alignment_Tool = QtWidgets.QMainWindow()
    ui = Ui_VIO_alignment_Tool()
    ui.setupUi(VIO_alignment_Tool)
    VIO_alignment_Tool.show()
    sys.exit(app.exec_())
