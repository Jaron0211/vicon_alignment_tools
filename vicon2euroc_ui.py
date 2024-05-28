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

#variable
best_offset = 0
best_error_sum = 9999999

####----------------------------------####

import pandas as pd
import numpy as np


#Make plot quicker
import matplotlib as mpl
import matplotlib.style as mplstyle
mplstyle.use(['fast'])
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['agg.path.chunksize'] = 1

from matplotlib.animation import FuncAnimation

from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
import time

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use("Qt5Agg")      # 表示使用 Qt5
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from main_ui import Ui_VIO_alignment_Tool

from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import axes3d

####--------------UI-----------------####

####--------------UI-----------------####

# vio_data = pd.read_csv(vio_file, header=None, names=['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz'])
# gt_data  = pd.read_csv(gt_file , header=None, names=['timestamp', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'])

# force finding, maybe change to the DTW method
# https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html

class Csv_Manager():

    def __init__(self, csv_file):

        self.path_data = pd.read_csv(csv_file, header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])

        self.name = csv_file.split('/')[-1].split('.')[0]

        self.shift = 0
        self.cache_data = self.path_data.copy()
        self.y_scale = 1.0
        self.x_scale = 1.0

        self.z_rotate = 0
        self.y_rotate = 0
        self.x_rotate = 0
    
    def save_modify_csv(self, start_time, end_time):

        self.cache_data[self.cache_data['timestamp'].between(start_time, end_time, inclusive="both")]

        self.cache_data.to_csv("Aligned_{}.csv".format(self.name) ,index=False, header=None, float_format='%.9f')

class Main(QtWidgets.QMainWindow, Ui_VIO_alignment_Tool):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.cwd = os.getcwd()

        self.gt_file = ''
        self.vio_file = ''

        self.gt_data = pd.DataFrame([[0,0,0,0,0,0,0,0]],columns=['timestamp', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'])
        self.vio_data = pd.DataFrame([[0,0,0,0,0,0,0,0,0,0,0]],columns=['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz'])
        self.vio_rotated = self.vio_data.copy()

        self.vio_presave = self.vio_data.copy()
        self.gt_presave = self.gt_data.copy()

        self.vio_presave_cache = self.vio_data.copy()
        self.gt_presave_cache = self.gt_data.copy()

        self.gt_E2Q_presave = self.vio_data.copy()

        self.csv_dict = {}

        self.rotation_angle = 0

        self.GT_file.clicked.connect(self.open_gt_file)
        self.VIO_file.clicked.connect(self.open_vio_file)

        self.canvas = FigureCanvas(plt.Figure())
        self.canvas_3d = FigureCanvas(plt.Figure())
        self.canvas.figure.set_facecolor("None")
        self.canvas.setStyleSheet("background-color:transparent;")
        self.canvas_layout.addWidget(NavigationToolbar(self.canvas, self))
        self.canvas_3d.figure.set_facecolor("None")
        self.canvas_3d.setStyleSheet("background-color:transparent;")
        self.canvas.setMinimumSize(QtCore.QSize(200, 300))

        self.canvas_3d.ax = self.canvas_3d.figure.add_subplot(122, projection='3d')
        self.canvas_3d.perspection = self.canvas_3d.figure.add_subplot(121)
        self.canvas.ax = self.canvas.figure.add_subplot(111)

        self.x_scale = 1
        self.y_scale = 1
        self.offset = 0
        self.range = 100
        self.VIO_shift = 0
        self.gt_shift = 0
        self.start_from_index = 0
        self.end_to_index = 1.7*10**18

        self.canvas_3d_layout.addWidget(self.canvas_3d, 0, 0, 1, 1)
        self.canvas_layout.addWidget(self.canvas)

        self.yscale_spinbox.valueChanged.connect(self.update_y_scale)
        self.xscale_spinbox.valueChanged.connect(self.update_x_scale)

        self.range_spinbox.valueChanged.connect(self.update_range)
        self.range_spinbox.setValue(1000)

        self.nav_offset_spinbox.valueChanged.connect(self.update_VIO_shift)
        self.offset_spinbox.valueChanged.connect(self.update_GT_shift)

        self.save_result.clicked.connect(self.save_alignment)

        self.nav_offset_spinbox.setMaximum(10000)
        self.offset_spinbox.setMaximum(10000)

        self.Start_from.valueChanged.connect(self.update_start_from)
        self.End_to.valueChanged.connect(self.update_end_to)

        self.Zangle_spinbox.valueChanged.connect(self.process_data)
        self.Yangle_spinbox.valueChanged.connect(self.process_data)
        self.Xangle_spinbox.valueChanged.connect(self.process_data)

        self.listWidget_CsvList.itemClicked.connect(self.update_frame)
        self.listWidget_CsvList.itemDoubleClicked.connect(self.remove_item)

        self.dial_vio_shift.valueChanged.connect(self.vio_shift_update)
        self.dial_gt_shift.valueChanged.connect(self.gt_shift_update)

        self.best_offset = 0
        self.best_error_sum = 9999999

        self.x_scale = self.xscale_spinbox.value()
        self.y_scale = self.yscale_spinbox.value()
        self.offset = self.offset_spinbox.value()
        self.range = self.range_spinbox.value()
        self.range_spinbox.setMaximum(100)
        self.VIO_shift = self.nav_offset_spinbox.value()
        self.gt_shift = self.cal_vio_inter_value(0)

        self.update_y_scale(1)
        self.update_x_scale(1)

    def vio_shift_update(self, float):
        return
    
    def gt_shift_update(self, float):
        return


    def open_gt_file(self):

        gt_file, filetype = QtWidgets.QFileDialog.getOpenFileName(self,  
                                    "Choose the GT path file",  
                                    self.cwd, # 起始路径 
                                    "All Files (*);;Text Files (*.txt)")
        
        _name = gt_file.split('/')[-1]
        if gt_file == "" or (_name in self.csv_dict):
            return
        
        self.csv_dict[_name] = Csv_Manager(gt_file)

        name = gt_file.split('/')[-1]

        self.listWidget_CsvList.addItem(name)
        self.listWidget_CsvList.findItems(name, Qt.MatchExactly)[0].setForeground(Qt.red)
        
        self.process_data()

    def open_vio_file(self):

        #self.ani_1.pause()
        #self.ani_2.pause()
        vio_file, filetype = QtWidgets.QFileDialog.getOpenFileName(self,  
                                    "Choose the VIO path file",  
                                    self.cwd, # 起始路径 
                                    "All Files (*);;Text Files (*.txt)") 

        #self.ani_1.resume()
        #self.ani_2.resume()
        _name = vio_file.split('/')[-1]
        if vio_file == "" or (_name in self.csv_dict):
            return
        
        self.csv_dict[_name] = Csv_Manager(vio_file)
        name = vio_file.split('/')[-1]

        self.listWidget_CsvList.addItem(name)
        self.listWidget_CsvList.findItems(name, Qt.MatchExactly)[0].setForeground(Qt.gray)

    def remove_item(self, item):

        target = item.text()
        self.csv_dict.pop(target)
        self.listWidget_CsvList.takeItem(self.listWidget_CsvList.currentRow())
        self.process_data()

    def update_frame(self, item):
        
        target = item.text()
        csv_class = self.csv_dict[target]

        self.Zangle_spinbox.setValue(csv_class.z_rotate)
        self.Xangle_spinbox.setValue(csv_class.y_rotate)
        self.Yangle_spinbox.setValue(csv_class.x_rotate)

        self.xscale_spinbox.setValue(csv_class.x_scale)
        self.yscale_spinbox.setValue(csv_class.y_scale)


    def update_x_scale(self, value):
        self.x_scale = value  # Mapping slider value to a reasonable range
        self.process_data()

    def update_y_scale(self, value):
        self.y_scale = value  # Mapping slider value to a reasonable range
        self.process_data()
    
    def update_offset(self, value):
        self.offset = value
        self.process_data()

    def update_range(self, value):
        self.range = self.cal_vio_inter_value(value*100)-min(self.vio_data['timestamp'])
        self.process_data()
    
    def update_VIO_shift(self, value):
        self.VIO_shift = value

        self.nav_offset_spinbox.setValue(value)
        self.vio_shifter.setValue(value)
        self.process_data()

    def cal_vio_inter_value(self, value):
        
        try:
            max(self.vio_data['timestamp'])
        except Exception:
            return 0
        return (value/10000 * (max(self.vio_data['timestamp']) - min(self.vio_data['timestamp'])) + min(self.vio_data['timestamp']))

    def update_GT_shift(self, value):

        self.gt_shift = self.cal_vio_inter_value(value)

        self.offset_spinbox.setValue(value)
        self.gt_shifter.setValue(value)

        self.start_from_index = self.cal_vio_inter_value(self.Start_from.value()) + self.gt_shift
        self.end_to_index = max(self.vio_data['timestamp']) - self.cal_vio_inter_value(self.End_to.value())
        self.process_data()

    def update_start_from(self, value):
        self.Start_from.setMaximum(10000)
        self.Start_from.setMinimum(0)

        new_index = self.cal_vio_inter_value(value)

        if new_index < self.end_to_index:
            self.start_from_index = new_index
        else:
            self.start_from_index = self.end_to_index
        self.process_data()

    def update_end_to(self, value):
        self.End_to.setMaximum(10000)
        self.End_to.setMinimum(0)

        new_index = max(self.vio_data['timestamp']) + self.gt_shift - self.cal_vio_inter_value(value)

        if new_index > self.start_from_index:
            self.end_to_index = new_index
        else:
            self.end_to_index = self.start_from_index
        self.process_data()

    def process_data(self):

        self.vio_presave_cache = self.vio_data.copy()
        self.gt_presave_cache  = self.gt_data.copy()

        #VIO rotate
        rotated_coordinates_xyz = np.column_stack((self.vio_data['px'], self.vio_data['py'], self.vio_data['pz']))
        rotation_xyz = R.from_euler('ZYX', [self.Zangle_spinbox.value(),self.Yangle_spinbox.value(),self.Xangle_spinbox.value()], degrees=True)
        rotated_coordinates_xyz = rotation_xyz.apply(rotated_coordinates_xyz)
        
        self.vio_presave_cache[['px','py','pz']] = rotated_coordinates_xyz

        #GT re-scale
        self.gt_presave_cache['timestamp']  = [ (self.vio_presave_cache['timestamp'][0] + (i)/self.x_scale*10**8 - self.VIO_shift*10**7) for i in range(0, len(self.gt_data['timestamp']), 1)]
        self.gt_presave_cache[['px','py','pz']] *= self.y_scale

        self.plot_data()
        self.Td_plot_data()

    def plot_data(self, num = 0):
        ## 2D timimg
        
        self.canvas.ax.clear()
        legend = []

        for name, csv in self.csv_dict.items():
        
            self.canvas.ax.scatter(csv.cache_data['timestamp'], csv.cache_data['px'], marker='o', s=1)
            legend.append(name)

        self.canvas.ax.legend(legend)
        self.canvas.ax.set_xlabel('Time')
        self.canvas.ax.set_ylabel('Position')
        self.canvas.ax.grid(True)   
        self.canvas.figure.tight_layout()
        self.canvas.ax.relim()
        self.start_from_index, self.end_to_index = self.canvas.ax.get_xlim()
        self.canvas.ax.autoscale_view()
        self.canvas.ax.axvline(x = self.start_from_index, color = 'r')
        self.canvas.ax.axvline(x = self.end_to_index, color = 'b')
        self.canvas.draw()

        return [self.canvas.ax]
    
    def Td_plot_data(self, num = 0):

        ## 3D R|T
        self.canvas_3d.ax.clear()
        #self.canvas_3d.ax.disable_mouse_rotation()
        self.canvas_3d.ax.plot(self.gt_presave_cache['px'].to_numpy(), self.gt_presave_cache['py'].to_numpy(), self.gt_presave_cache['pz'].to_numpy(), color = 'r', label='Groundtruth Path')
        self.canvas_3d.ax.scatter(self.vio_presave_cache['px'].to_numpy(), self.vio_presave_cache['py'].to_numpy(), self.vio_presave_cache['pz'].to_numpy(),color = 'g', s=.3, label='VIO Path')

        self.canvas_3d.ax.set_xlabel('X(m)')
        self.canvas_3d.ax.set_ylabel('Y(m)')
        self.canvas_3d.ax.set_zlabel('Z(m)')
        self.canvas_3d.ax.legend(['Groundtruth Path', "VIO Path"])
        self.canvas_3d.ax.set_facecolor("None")

        self.canvas_3d.perspection.clear()
        self.canvas_3d.perspection.grid(True)
        self.canvas_3d.perspection.set_xlabel('X(m)')
        self.canvas_3d.perspection.set_ylabel('Y(m)')
        self.canvas_3d.perspection.plot(self.gt_presave_cache['px'].to_numpy(), self.gt_presave_cache['py'].to_numpy(), color = 'r')
        self.canvas_3d.perspection.plot(self.vio_presave_cache['px'].to_numpy(), self.vio_presave_cache['py'].to_numpy(), color = 'g')
        
        self.canvas_3d.figure.tight_layout()
        self.canvas_3d.draw()
        return [self.canvas_3d.ax, self.canvas_3d.perspection]
    
    def rotated_figure(self, num):

        self.rotation_angle = self.canvas_3d.ax.azim + 1
        # Normalize the angle to the range [-180, 180] for display
        angle_norm = (self.rotation_angle + 180) % 360 - 180

        # Cycle through a full rotation of elevation, then azimuth, roll, and all
        # elev = azim = roll = 0
        azim = 0
        if self.rotation_angle <= 360:
            azim = self.rotation_angle
        else:
            azim = angle_norm

        self.canvas_3d.ax.view_init(self.canvas_3d.ax.elev, azim)
        print(self.rotation_angle)

        return [self.canvas_3d.ax]

    def save_alignment(self):

        self.vio_presave = self.vio_presave_cache[self.vio_presave_cache['timestamp'].between(self.start_from_index,self.end_to_index, inclusive="both")]
        self.gt_presave = self.gt_presave_cache[self.gt_presave_cache['timestamp'].between(self.vio_presave['timestamp'].min(),self.vio_presave['timestamp'].max(), inclusive="both")]

        self.vio_presave.to_csv('timed_vio_crop.csv',index=False, header=None, float_format='%.9f')
        self.gt_presave.to_csv('timed_gt_crop.csv'  ,index=False, header=None, float_format='%.9f')

        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    window = Main()
    window.show()
    sys.exit(app.exec_())