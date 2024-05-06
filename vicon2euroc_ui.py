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

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use("Qt5Agg")      # 表示使用 Qt5
import matplotlib.pyplot as plt

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
import sys
import os
import time

from main_ui import Ui_VIO_alignment_Tool

from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import axes3d

####--------------UI-----------------####
from PyQt5 import QtCore, QtGui, QtWidgets

####--------------UI-----------------####

vio_data = pd.read_csv(vio_file, header=None, names=['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz'])
gt_data  = pd.read_csv(gt_file , header=None, names=['timestamp', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'])

# force finding, maybe change to the DTW method
# https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html

class Main(QtWidgets.QMainWindow, Ui_VIO_alignment_Tool):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.cwd = os.getcwd()

        self.GT_file.clicked.connect(self.open_gt_file)
        self.VIO_file.clicked.connect(self.open_vio_file)

        self.canvas = FigureCanvas(plt.Figure())
        self.canvas_3d = FigureCanvas(plt.Figure())
        self.gridLayout.addWidget(self.canvas, 5, 0, 8, 2)
        self.gridLayout.addWidget(self.canvas_3d, 0, 0, 5, 3)

        self.yscale_spinbox.valueChanged.connect(self.update_y_scale)
        self.xscale_spinbox.valueChanged.connect(self.update_x_scale)

        self.range_spinbox.valueChanged.connect(self.update_range)
        self.range_spinbox.setValue(1000)

        self.vio_shifter.valueChanged.connect(self.update_VIO_shift)
        self.gt_shifter.valueChanged.connect(self.update_GT_shift)

        self.nav_offset_spinbox.valueChanged.connect(self.update_VIO_shift)
        self.offset_spinbox.valueChanged.connect(self.update_GT_shift)

        self.save_result.clicked.connect(self.save_alignment)

        self.vio_shifter.setMaximum(10000)
        self.gt_shifter.setMaximum(10000)

        self.nav_offset_spinbox.setMaximum(10000)
        self.offset_spinbox.setMaximum(10000)

        self.Start_from.valueChanged.connect(self.update_start_from)
        self.End_to.valueChanged.connect(self.update_end_to)

        self.gt_file = gt_file
        self.vio_file = vio_file

        self.gt_data = gt_data
        self.vio_data = vio_data
        self.vio_rotated = self.vio_data.copy()

        self.x_scale = self.xscale_spinbox.value()
        self.y_scale = self.yscale_spinbox.value()
        self.offset = self.offset_spinbox.value()
        self.range = self.range_spinbox.value()
        self.range_spinbox.setMaximum(100)
        self.VIO_shift = self.nav_offset_spinbox.value()
        self.gt_shift = self.cal_vio_inter_value(0)

        self.best_offset = 0
        self.best_error_sum = 9999999

        self.canvas_3d.ax = self.canvas_3d.figure.add_subplot(122, projection='3d')
        self.canvas_3d.perspection = self.canvas_3d.figure.add_subplot(121)
        self.canvas.ax = self.canvas.figure.add_subplot(111)

        self.start_from_index = 0
        self.end_to_index = 1.7*10**18

        self.update_y_scale(1)
        self.update_x_scale(1)

        self.ani_1 = FuncAnimation(self.canvas_3d.figure, self.Td_plot_data, frames=np.arange(0,360), repeat=1, interval=100, blit=True)  
        self.ani_2 = FuncAnimation(self.canvas.figure, self.plot_data, frames=np.arange(0,1), repeat=1, interval=10, blit=True)  
    
    def open_gt_file(self):
        self.gt_file, filetype = QtWidgets.QFileDialog.getOpenFileName(self,  
                                    "Choose the GT path file",  
                                    self.cwd, # 起始路径 
                                    "All Files (*);;Text Files (*.txt)")   # 设置文件扩展名过滤,用双分号间隔

        if self.gt_file == "":
            return
        
        global gt_data
        self.gt_data = pd.read_csv(self.gt_file, header=None, names=['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz'])

    def open_vio_file(self):
        self.vio_file, filetype = QtWidgets.QFileDialog.getOpenFileName(self,  
                                    "Choose the VIO path file",  
                                    self.cwd, # 起始路径 
                                    "All Files (*);;Text Files (*.txt)")   # 设置文件扩展名过滤,用双分号间隔

        if self.vio_file == "":
            return
        
        global vio_data
        vio_data = pd.read_csv(self.vio_file, header=None, names=['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz'])
        self.end_to_index = len(self.vio_data['timestamp'])

    def update_x_scale(self, value):
        self.x_scale = value  # Mapping slider value to a reasonable range

    def update_y_scale(self, value):
        self.y_scale = value  # Mapping slider value to a reasonable range
    
    def update_offset(self, value):
        self.offset = value

    def update_range(self, value):
        self.range = self.cal_vio_inter_value(value*100)-min(self.vio_data['timestamp'])
    
    def update_VIO_shift(self, value):
        self.VIO_shift = value

        self.nav_offset_spinbox.setValue(value)
        self.vio_shifter.setValue(value)

    def cal_vio_inter_value(self, value):
        return (value/10000 * (max(self.vio_data['timestamp']) - min(self.vio_data['timestamp'])) + min(self.vio_data['timestamp']))

    def update_GT_shift(self, value):

        self.gt_shift = self.cal_vio_inter_value(value)

        self.offset_spinbox.setValue(value)
        self.gt_shifter.setValue(value)

        self.start_from_index = self.cal_vio_inter_value(self.Start_from.value()) + self.gt_shift
        self.end_to_index = max(self.vio_data['timestamp']) - self.cal_vio_inter_value(self.End_to.value())

    def update_start_from(self, value):
        self.Start_from.setMaximum(10000)
        self.Start_from.setMinimum(0)

        new_index = self.cal_vio_inter_value(value)

        if new_index < self.end_to_index:
            self.start_from_index = new_index
        else:
            self.start_from_index = self.end_to_index


    def update_end_to(self, value):
        self.End_to.setMaximum(10000)
        self.End_to.setMinimum(0)

        new_index = max(self.vio_data['timestamp']) + self.gt_shift - self.cal_vio_inter_value(value)

        if new_index > self.start_from_index:
            self.end_to_index = new_index
        else:
            self.end_to_index = self.start_from_index

    # def plot_data(self, num):
    #     ## 2D timimg
    #     vio_compare_array = self.vio_rotated['px'] * self.y_scale

    #     gt_indexs = [i for i in range(0, len(gt_data['px']), int(self.x_scale))]
    #     gt_compare_array = self.gt_data['px'][gt_indexs]

    #     self.canvas.ax.clear()
    #     self.canvas.ax.scatter(range(0, len(gt_compare_array), 1), gt_compare_array, color='blue',marker='o',s=.1)
    #     self.canvas.ax.scatter(range(self.gt_shift, len(vio_compare_array) + self.gt_shift, 1), vio_compare_array,color='red',marker='o',s=.1)

    #     self.canvas.ax.set_xlabel('Time')
    #     self.canvas.ax.set_ylabel('Position')
    #     self.canvas.ax.legend([self.gt_file, self.vio_file])
    #     self.canvas.ax.set_title('VIO Path vs Vicon GT')

    #     self.canvas.ax.set_xlim([self.VIO_shift,self.VIO_shift + self.range])
    #     self.canvas.ax.axvline(x = self.start_from_index, color = 'r')
    #     self.canvas.ax.axvline(x = self.end_to_index, color = 'b')

    #     return [self.canvas.ax]

    def plot_data(self, num):
        ## 2D timimg

        vio_compare_index = self.vio_rotated['timestamp']
        vio_compare_array = self.vio_rotated['px'] * self.y_scale

        gt_indexs = [ (self.vio_rotated['timestamp'][0] + (i)/self.x_scale*10**8 - self.VIO_shift*10**7) for i in range(0, len(gt_data['timestamp']), 1)]
        gt_compare_array = self.gt_data['px']

        self.canvas.ax.clear()
        self.canvas.ax.scatter(gt_indexs, gt_compare_array, color='blue',marker='o',s=.1)
        self.canvas.ax.scatter(vio_compare_index, vio_compare_array,color='red',marker='o',s=.1)

        self.canvas.ax.set_xlabel('Time')
        self.canvas.ax.set_ylabel('Position')
        self.canvas.ax.legend([self.gt_file, self.vio_file])
        self.canvas.ax.set_title('VIO Path vs Vicon GT')

        self.canvas.ax.set_xlim([self.gt_shift, self.gt_shift+self.range])
        self.canvas.ax.axvline(x = self.start_from_index, color = 'r')
        self.canvas.ax.axvline(x = self.end_to_index, color = 'b')

        return [self.canvas.ax]
    
    def Td_plot_data(self, num):

        ## 3D R|T
        rotated_coordinates_xyz = np.column_stack((self.vio_data['px'], self.vio_data['py'], self.vio_data['pz']))
        rotation_xyz = R.from_euler('ZYX', [self.yscale_spinbox_2.value(),self.yscale_spinbox_3.value(),self.yscale_spinbox_4.value()], degrees=True)
        rotated_coordinates_xyz = rotation_xyz.apply(rotated_coordinates_xyz)

        self.vio_rotated = self.vio_data.copy()
        self.vio_rotated[['px','py','pz']] = rotated_coordinates_xyz

        self.canvas_3d.ax.clear()
        self.canvas_3d.ax.disable_mouse_rotation()
        self.canvas_3d.ax.plot(self.gt_data['px'], self.gt_data['py'], self.gt_data['pz'], color = 'r', label='Origin Data')
        self.canvas_3d.ax.scatter(self.vio_rotated['px'], self.vio_rotated['py'], self.vio_rotated['pz'], s=.5, label='Rotated Data')

        self.canvas_3d.ax.set_xlabel('X(m)')
        self.canvas_3d.ax.set_ylabel('Y(m)')
        self.canvas_3d.ax.set_zlabel('Z(m)')
        self.canvas_3d.ax.legend([self.gt_file, self.vio_file])
        self.canvas_3d.ax.set_title('VIO Path vs Vicon GT')

        self.canvas_3d.perspection.clear()
        self.canvas_3d.perspection.plot(self.gt_data['px'], self.gt_data['py'])
        self.canvas_3d.perspection.plot(self.vio_rotated['px'], self.vio_rotated['py'])

        angle = num
        # Normalize the angle to the range [-180, 180] for display
        angle_norm = (angle + 180) % 360 - 180

        # Cycle through a full rotation of elevation, then azimuth, roll, and all
        # elev = azim = roll = 0
        azim = 0
        if angle <= 360:
            azim = angle
        #     elev = angle_norm
        # elif angle <= 360*2:
        #     azim = angle_norm
        # elif angle <= 360*3:
        #     roll = angle_norm
        else:
            azim = angle_norm

        self.canvas_3d.ax.view_init(30, azim)

        return [self.canvas_3d.ax, self.canvas_3d.perspection]

    def Td_rotation(self):
        

        angle = 0
        # Normalize the angle to the range [-180, 180] for display
        angle_norm = (angle + 180) % 360 - 180

        # Cycle through a full rotation of elevation, then azimuth, roll, and all
        # elev = azim = roll = 0
        azim = 0
        if angle <= 360:
            azim = angle
        #     elev = angle_norm
        # elif angle <= 360*2:
        #     azim = angle_norm
        # elif angle <= 360*3:
        #     roll = angle_norm
        else:
            azim = angle_norm

        # Update the axis view and title
        #print(elev, azim, roll)
        
        self.canvas_3d.ax.view_init(90, azim)

        return self.canvas_3d.ax


    def save_alignment(self):

        gt_start = self.offset
        vio_end = int((len(gt_data['px'])-gt_start)/self.x_scale)

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

        vio_data_crop.to_csv('timed_dgvins_vio_crop.csv',index=False, header=None, float_format='%.9f')
        gt_data_crop.to_csv('timed_gt_crop.csv'         ,index=False, header=None, float_format='%.9f')
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