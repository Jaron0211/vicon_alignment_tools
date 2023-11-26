'''
Date: 20231124
Author: Jian-Lun, Li
Filename: vicon2euroc.py
Describe:
This file cropping the vio_path and vicon_gt file into same size, and labeling the vio_path time into vicon_gt file.

'''

####---------parameters----------####

# file name
vio_file = './csv/dgvins_vio.csv'
gt_file  = './csv/data.csv'

# adjustment
x_scale = 10
y_scale = 1000
offset = 0

#variable
best_offset = 0
best_error_sum = 9999999

####----------------------------------####

import pandas as pd
import numpy as np

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use("Qt5Agg")      # 表示使用 Qt5
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

from main_ui import Ui_MainWindow

from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import axes3d

####--------------UI-----------------####
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(545, 481)
        MainWindow.setMinimumSize(QtCore.QSize(545, 481))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)
        self.xscale_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.xscale_spinBox.setMinimum(1)
        self.xscale_spinBox.setMaximum(10000)
        self.xscale_spinBox.setObjectName("xscale_spinBox")
        self.gridLayout.addWidget(self.xscale_spinBox, 3, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 6, 2, 1, 1)
        self.offset_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.offset_spinBox.setMinimum(1)
        self.offset_spinBox.setMaximum(10000)
        self.offset_spinBox.setObjectName("offset_spinBox")
        self.gridLayout.addWidget(self.offset_spinBox, 5, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 4, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 2, 1, 1)
        self.yscale_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.yscale_spinBox.setMinimum(1)
        self.yscale_spinBox.setMaximum(10000)
        self.yscale_spinBox.setObjectName("yscale_spinBox")
        self.gridLayout.addWidget(self.yscale_spinBox, 1, 2, 1, 1)
        self.range_spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.range_spinBox.setMinimum(100)
        self.range_spinBox.setMaximum(10000)
        self.range_spinBox.setObjectName("range_spinBox")
        self.gridLayout.addWidget(self.range_spinBox, 7, 2, 1, 1)
        self.save_button = QtWidgets.QPushButton(self.centralwidget)
        self.save_button.setText('Save result')
        self.save_button.setObjectName("save_botton")
        self.gridLayout.addWidget(self.save_button, 8, 2, 1, 1)
        self.canvas = FigureCanvas(plt.Figure())
        self.canvas.setMinimumSize(QtCore.QSize(400, 400))
        self.canvas.setObjectName("canvas")
        self.gridLayout.addWidget(self.canvas, 0, 0, 8, 2)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 545, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "Y_scale"))
        self.label_4.setText(_translate("MainWindow", "Show range"))
        self.label_3.setText(_translate("MainWindow", "Offset"))
        self.label.setText(_translate("MainWindow", "X scale"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))



####--------------UI-----------------####

vio_data = pd.read_csv(vio_file, header=None, names=['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz'])
gt_data  = pd.read_csv(gt_file , header=None, names=['timestamp', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'])

# force finding, maybe change to the DTW method
# https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html

class Main(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.yscale_spinBox.valueChanged.connect(self.update_y_scale)
        self.xscale_spinBox.valueChanged.connect(self.update_x_scale)
        self.offset_spinBox.valueChanged.connect(self.update_offset)
        self.range_spinBox.valueChanged.connect(self.update_range)
        self.save_button.clicked.connect(self.save_alignment)

        self.x_scale = x_scale
        self.y_scale = y_scale
        self.offset = offset
        self.range = 50

        self.best_offset = 0
        self.best_error_sum = 9999999

        self.canvas.ax = self.canvas.figure.add_subplot(111)

        self.plot_data()
        self.update_y_scale(1)
        self.update_x_scale(1)
        
    def update_x_scale(self, value):
        self.x_scale = value  # Mapping slider value to a reasonable range
        self.plot_data()

    def update_y_scale(self, value):
        self.y_scale = value  # Mapping slider value to a reasonable range
        self.plot_data()
    
    def update_offset(self, value):
        self.offset = value
        self.plot_data()

    def update_range(self, value):
        self.range = value
        self.plot_data()

    def plot_data(self):
        vio_compare_array = vio_data['px'][0:self.range] * self.y_scale

        gt_indexs = [i for i in range(self.offset, self.offset + int(self.x_scale * self.range), int(self.x_scale))]
        gt_compare_array = gt_data['py'][gt_indexs]

        self.canvas.ax.clear()
        self.canvas.ax.plot(range(0, len(gt_compare_array), 1), gt_compare_array)
        self.canvas.ax.plot(range(0, len(vio_compare_array), 1), vio_compare_array)
        
        self.canvas.ax.set_xlabel('Time')
        self.canvas.ax.set_ylabel('Position')
        self.canvas.ax.legend(['VIO Path', 'Vicon GT'])
        self.canvas.ax.set_title('VIO Path vs Vicon GT')

        self.canvas.draw()

    def save_alignment(self):

        gt_start = self.offset
        vio_end = int((len(gt_data['py'])-gt_start)/self.x_scale)

        vio_data_crop = vio_data[:vio_end]
        gt_data_crop  = gt_data[gt_start:]

        time_start = vio_data_crop['timestamp'][ 0]
        time_end   = vio_data_crop['timestamp'][len(vio_data_crop['timestamp'])-1]

        time_spend = time_end - time_start
        len_of_vio = len(gt_data_crop['timestamp'])
        time_space = int(time_spend / len_of_vio)

        gt_data_crop['timestamp'][0] = time_start
        gt_data_crop['timestamp'][len(gt_data_crop['timestamp'])-1] = time_end

        timing_array = [i for i in range(int(time_start), int(time_end) ,time_space)]
        gt_data_crop['timestamp'] = timing_array[:len(gt_data_crop['timestamp'])]

        vio_data_crop.to_csv('timed_dgvins_vio.csv',index=False, header=None, float_format='%.9f')
        gt_data_crop.to_csv('timed_gt.csv'         ,index=False, header=None, float_format='%.9f')
        print('ok')
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    window = Main()
    window.show()
    sys.exit(app.exec_())

# # Show the aligned result
# fig2 = plt.figure('alignment_result')
# cx = fig2.add_subplot(111)

# gt_start = 1322
# vio_end = int((len(gt_data['py'])-gt_start)/x_scale)
# print(vio_end)

# best_offset = 1322
# cx.plot(range(best_offset,best_offset+len(vio_data['px'][:vio_end])*x_scale,x_scale),vio_data['px'][:1224]*y_scale)
# cx.plot(gt_data['py'][1322:])
# plt.show()

# vio_data_crop = vio_data[:1224]
# gt_data_crop  = gt_data[1322:]

# time_start = vio_data_crop['timestamp'][ 0]
# time_end   = vio_data_crop['timestamp'][len(vio_data_crop['timestamp'])-1]

# print(time_start, time_end)

# time_spend = time_end - time_start
# len_of_vio = len(gt_data_crop['timestamp'])
# time_space = int(time_spend / len_of_vio)

# gt_data_crop['timestamp'][0] = time_start
# gt_data_crop['timestamp'][len(gt_data_crop['timestamp'])-1] = time_end

# timing_array = [i for i in range(int(time_start), int(time_end) ,time_space)]
# gt_data_crop['timestamp'] = timing_array[:len(gt_data_crop['timestamp'])]


# vio_data_crop.to_csv('timed_dgvins_vio.csv',index=False, header=None, float_format='%.9f')
# gt_data_crop.to_csv('timed_gt.csv',index=False, header=None, float_format='%.9f')