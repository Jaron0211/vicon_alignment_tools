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
import asyncio
import timeit

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use("Qt5Agg")      # 表示使用 Qt5
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from main_ui import Ui_VIO_alignment_Tool

from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import axes3d

import multiprocessing, threading

####--------------UI-----------------####

####--------------UI-----------------####

# vio_data = pd.read_csv(vio_file, header=None, names=['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz'])
# gt_data  = pd.read_csv(gt_file , header=None, names=['timestamp', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'])

# force finding, maybe change to the DTW method
# https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html

class Csv_Manager():

    def __init__(self, csv_file):

        self.path_data = pd.read_csv(csv_file, header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])

        self.name = csv_file.split('/')[-1]
        self.len = len(self.path_data['timestamp'])
        
        self.shift = 0
        self.cache_data = self.path_data.copy()

        self.interpolation_timestamp = pd.Series()

        self.y_scale = 1.0
        self.x_scale = 1.0

        self.y_scale_pre = self.y_scale
        self.x_scale_pre = self.x_scale

        self.z_rotate = 0
        self.y_rotate = 0
        self.x_rotate = 0

        self.delta_time = self.path_data['timestamp'].diff()
    
    def save_modify_csv(self, start_time, end_time):

        self.cache_data[self.cache_data['timestamp'].between(start_time, end_time, inclusive="both")]
        self.cache_data['timestamp'] = self.cache_data['timestamp'].astype(float)  
        self.cache_data.to_csv("Aligned_{}.csv".format(self.name) ,index=False, header=None, float_format='%.9f')
        
    def process_data(self):
        
        def job():
            
            start = time.time()
            rotated_coordinates_xyz = np.column_stack((self.path_data['px'], self.path_data['py'], self.path_data['pz']))
            rotation_xyz = R.from_euler('ZYX', [self.z_rotate,self.y_rotate,self.x_rotate], degrees=True)
            rotated_coordinates_xyz = rotation_xyz.apply(rotated_coordinates_xyz)
            
            self.cache_data[['px','py','pz']] = rotated_coordinates_xyz * self.y_scale

            _starttime = self.path_data['timestamp'][0]

            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

            def iterrator(i):
                self.interpolation_timestamp[i] = self.interpolation_timestamp[i-1] + self.delta_time[i] * self.x_scale 

            def looper():
                self.interpolation_timestamp[0] = _starttime + self.shift
                [iterrator(i) for i in range(1, len(self.path_data['timestamp']),1)]
            
            looper()

            self.cache_data['timestamp'] = self.interpolation_timestamp 

            print(self.name , ' time spend: ', time.time() - start)
            

        worker = threading.Thread(target=job)
        worker.start()

    def create_timestamp_via_Hz(self, start_time, hz = 100):

        self.path_data['timestamp'] = [ (start_time + (i)/self.x_scale*10**6) for i in range(0, self.len, 1)]
        self.delta_time = self.path_data['timestamp'].diff()


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
        self.current_item = ''
        self.gt_first_timestamp = 0

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

        #self.nav_offset_spinbox.valueChanged.connect(self.update_VIO_shift)
        #self.offset_spinbox.valueChanged.connect(self.update_GT_shift)

        self.save_result.clicked.connect(self.save_alignment)

        self.Start_from.sliderMoved.connect(self.update_start_from_and_end_to)
        self.Start_from.setMaximum(10000)
        self.Start_from.setMinimum(0)

        self.End_to.sliderMoved.connect(self.update_start_from_and_end_to)
        self.End_to.setMaximum(10000)
        self.End_to.setMinimum(0)

        self.Zangle_spinbox.valueChanged.connect(self.update_rotation)
        self.Yangle_spinbox.valueChanged.connect(self.update_rotation)
        self.Xangle_spinbox.valueChanged.connect(self.update_rotation)

        self.listWidget_CsvList.itemClicked.connect(self.update_frame)
        self.listWidget_CsvList.itemDoubleClicked.connect(self.remove_item)

        self.dial_vio_shift.valueChanged.connect(self.vio_shift_update)
        self.dial_gt_shift.valueChanged.connect(self.gt_shift_update)

        self.vio_shift_last = 0
        self.gt_shift_last = 0

        self.x_scale = self.xscale_spinbox.value()
        self.y_scale = self.yscale_spinbox.value()
        self.offset = self.offset_spinbox.value()
        self.range = self.range_spinbox.value()
        self.range_spinbox.setMaximum(100)
        self.VIO_shift = self.nav_offset_spinbox.value()
        self.gt_shift = self.cal_vio_inter_value(0)

        self.update_y_scale(1)
        self.update_x_scale(1)

        self.process_worker = threading.Thread(target=self.process_data_looper)
        self.process_worker.start()

    def process_data_looper(self):

        while 1:
            self.process_data()

    def vio_shift_update(self, value):

        if value - self.vio_shift_last > 0:
            self.offset_spinbox.setValue(self.offset_spinbox.value()+1)
        elif value - self.vio_shift_last < 0:
            self.offset_spinbox.setValue(self.offset_spinbox.value()-1)

        self.vio_shift_last = value
        return
    
    def gt_shift_update(self, value):
        
        print(value - self.gt_shift_last)
        if (value - self.gt_shift_last) > 0:
            self.nav_offset_spinbox.setValue(self.nav_offset_spinbox.value()+1)
            if self.gt_file != '' : 
                self.csv_dict[self.gt_file].shift += 100000000
                print(self.csv_dict[self.gt_file].shift)
        else:
            self.nav_offset_spinbox.setValue(self.nav_offset_spinbox.value()-1)
            if self.gt_file != '' : 
                self.csv_dict[self.gt_file].shift -= 100000000
                print(self.csv_dict[self.gt_file].shift)

        self.gt_shift_last = value
        return

    def update_gt_timestamp(self, timestamp):

        if type(timestamp) not in [ float , int ]:
            print('type error')
            return

        if self.gt_first_timestamp == 0 and self.gt_file != '':
            self.gt_first_timestamp = timestamp
            self.csv_dict[self.gt_file].create_timestamp_via_Hz(timestamp)

    def open_gt_file(self):

        gt_file, filetype = QtWidgets.QFileDialog.getOpenFileName(self,  
                                    "Choose the GT path file",  
                                    self.cwd, # 起始路径 
                                    "All Files (*);;Text Files (*.txt)")
        
        _name = gt_file.split('/')[-1]
        if gt_file == "" or (_name in self.csv_dict):
            return
        
        self.csv_dict[_name] = Csv_Manager(gt_file)

        self.listWidget_CsvList.addItem(_name)
        self.listWidget_CsvList.findItems(_name, Qt.MatchExactly)[0].setForeground(Qt.red)
        self.gt_file = _name

    def open_vio_file(self):

        vio_file, filetype = QtWidgets.QFileDialog.getOpenFileName(self,  
                                    "Choose the VIO path file",  
                                    self.cwd, # 起始路径 
                                    "All Files (*);;Text Files (*.txt)") 

        _name = vio_file.split('/')[-1]
        if vio_file == "" or (_name in self.csv_dict):
            return
        
        self.csv_dict[_name] = Csv_Manager(vio_file)
        self.listWidget_CsvList.addItem(_name)
        self.listWidget_CsvList.findItems(_name, Qt.MatchExactly)[0].setForeground(Qt.gray)
        self.update_gt_timestamp(min(self.csv_dict[_name].path_data['timestamp']))

    def remove_item(self, item):

        target = item.text()
        self.csv_dict.pop(target)
        self.listWidget_CsvList.takeItem(self.listWidget_CsvList.currentRow())

    def update_frame(self, item):
        
        target = item.text()
        self.current_item = target
        print(target)
        csv_class = self.csv_dict[target]

        self.Zangle_spinbox.setValue(csv_class.z_rotate)
        self.Xangle_spinbox.setValue(csv_class.y_rotate)
        self.Yangle_spinbox.setValue(csv_class.x_rotate)

        self.xscale_spinbox.setValue(csv_class.x_scale)
        self.yscale_spinbox.setValue(csv_class.y_scale)

    def update_x_scale(self, value):

        if self.current_item == '':
                return 
        
        self.csv_dict[self.current_item].x_scale = value  # Mapping slider value to a reasonable range

    def update_y_scale(self, value):

        if self.current_item == '':
                return
        
        self.csv_dict[self.current_item].y_scale = value

    def update_range(self, value):

        if self.current_item == '':
                return
        
        self.range = self.cal_vio_inter_value(value*100)-min(self.vio_data['timestamp'])

    def cal_vio_inter_value(self, value):
        
        try:
            max(self.vio_data['timestamp'])
        except Exception:
            return 0
        return (value/10000 * (max(self.vio_data['timestamp']) - min(self.vio_data['timestamp'])) + min(self.vio_data['timestamp']))

    def update_VIO_shift(self, value):
        if self.current_item == '':
                return
        
        self.csv_dict[self.current_item].shift = value

    def update_GT_shift(self, value):

        if self.gt_file == '':
                return
        
        self.csv_dict[self.gt_file].shift = value

    def update_end_to(self, value):
        

        new_index = self.cal_vio_inter_value(value)

        if new_index < self.end_to_index:
            self.start_from_index = new_index
        else:
            self.start_from_index = self.end_to_index

    def update_start_from_and_end_to(self, value):

        [x_min ,x_max ] = self.canvas.ax.get_xlim()

        self.start_from_index =  x_min + self.Start_from.value() /10000 * (x_max - x_min)
        self.end_to_index = x_max - self.End_to.value() /10000 * (x_max - x_min)

        if self.start_from_index > self.end_to_index:
            self.start_from_index = self.end_to_index

        print(self.start_from_index, self.end_to_index)

    def update_rotation(self):

        if self.current_item == '':
                return
        
        self.csv_dict[self.current_item].z_rotate = self.Zangle_spinbox.value()  # Mapping slider value to a reasonable range
        self.csv_dict[self.current_item].y_rotate = self.Yangle_spinbox.value()
        self.csv_dict[self.current_item].x_rotate = self.Xangle_spinbox.value()

    def process_data(self):

        [ threading.Thread(target = csv.process_data).start() for name, csv in self.csv_dict.items()]

        start = time.time()
        self.plot_data()
        self.Td_plot_data()

        print('plot spend: ' , time.time() - start)

    def plot_data(self, num = 0):
        ## 2D timimg
        def job():
            locker = threading.Lock()
            locker.acquire()
            
            self.canvas.ax.clear()
            legend = []

            min_val = [ min(value.cache_data['timestamp']) for _, value in self.csv_dict.items() if str(min(value.cache_data['timestamp'])) != 'nan']
            max_val = [ max(value.cache_data['timestamp']) for _, value in self.csv_dict.items() if str(max(value.cache_data['timestamp'])) != 'nan']

            if len(min_val) != 0:
                self.canvas.ax.set_xlim([min(min_val), max(max_val)])

            for name, csv in list(self.csv_dict.items()):
                if type(csv) != Csv_Manager:
                    continue 

                self.canvas.ax.plot(csv.cache_data['timestamp'][0::5], csv.cache_data['px'][0::5])
                legend.append(name)

            self.canvas.ax.legend(legend)
            self.canvas.ax.set_xlabel('Time')
            self.canvas.ax.set_ylabel('Position')
            self.canvas.ax.grid(True)   
            self.canvas.figure.tight_layout()
            
            self.canvas.ax.autoscale_view()
            self.canvas.ax.axvline(x = self.start_from_index, color = 'r')
            self.canvas.ax.axvline(x = self.end_to_index, color = 'b')
            self.canvas.draw()

            locker.release()

        worker = threading.Thread(target=job)
        worker.run()
    
    def Td_plot_data(self, num = 0):

        ## 3D R|T
        def job():

            locker = threading.Lock()
            locker.acquire()
            self.canvas_3d.ax.clear()
            self.canvas_3d.perspection.clear()
            legend = []

            for name, csv in self.csv_dict.copy().items():
                if type(csv) != Csv_Manager:
                    continue
                self.canvas_3d.ax.scatter(csv.cache_data['px'][0::5], csv.cache_data['py'][0::5], csv.cache_data['pz'][0::5], marker='o', s=1)
                self.canvas_3d.perspection.scatter(csv.cache_data['px'][0::5], csv.cache_data['py'][0::5], marker='o', s=1)
            
                legend.append(name)

            self.canvas_3d.ax.legend(legend)
            self.canvas_3d.ax.set_facecolor("None")
            self.canvas_3d.ax.set_xlabel('X(m)')
            self.canvas_3d.ax.set_ylabel('Y(m)')
            self.canvas_3d.ax.set_zlabel('Z(m)')
            self.canvas_3d.ax.legend(legend)

            
            self.canvas_3d.perspection.grid(True)
            self.canvas_3d.perspection.set_xlabel('X(m)')
            self.canvas_3d.perspection.set_ylabel('Y(m)')
            
            #self.canvas_3d.figure.tight_layout()
            self.canvas_3d.draw()
            locker.release()
 
        worker = threading.Thread(target=job)
        worker.run()
    
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

        for name, csv in self.csv_dict.items():
            if type(csv) != Csv_Manager:
                continue

            csv.save_modify_csv(self.start_from_index, self.end_to_index)

        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    window = Main()
    window.show()
    sys.exit(app.exec_())