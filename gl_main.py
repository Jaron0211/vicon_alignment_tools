from csv_manager import Csv_Manager

import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl

import pyqtgraph as pg
import math, threading
import time
import pandas as pd
import numpy as np
import os

uiclass, baseclass = pg.Qt.loadUiType("./ui/main.ui")    

class MainWindow(uiclass, baseclass):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.three_d_plot.show()
        self.three_d_plot.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.three_d_plot.setBackgroundColor((100,100,100,0))

        #variables
        self.cwd = os.getcwd()

        self.vio_shift_last = 0
        self.gt_shift_last = 0

        self.gt_file = ''

        self.csv_dict : dict[str, Csv_Manager] = {}
        self.current_item = ''
        self.gt_first_timestamp = 0

        self.rotation_angle = 0

        self.VIO_file.setEnabled(False)
        self.auto_align_button.setEnabled(False)
        #ui item

        self.GT_file.clicked.connect(self.open_gt_file)
        self.VIO_file.clicked.connect(self.open_vio_file)
        self.save_result.clicked.connect(self.save_alignment)

        self.Start_from.sliderMoved.connect(self.update_start_from_and_end_to)

        self.End_to.sliderMoved.connect(self.update_start_from_and_end_to)

        self.yscale_spinbox.valueChanged.connect(self.update_y_scale)
        self.xscale_spinbox.valueChanged.connect(self.update_x_scale)

        self.range_spinbox.valueChanged.connect(self.update_range)
        self.range_spinbox.setValue(1000)

        self.Zangle_spinbox.valueChanged.connect(self.update_rotation)
        self.Yangle_spinbox.valueChanged.connect(self.update_rotation)
        self.Xangle_spinbox.valueChanged.connect(self.update_rotation)

        self.listWidget_CsvList.itemClicked.connect(self.update_frame)
        self.listWidget_CsvList.itemDoubleClicked.connect(self.remove_item)

        self.dial_vio_shift.valueChanged.connect(self.vio_shift_update)
        self.dial_gt_shift.valueChanged.connect(self.gt_shift_update)

        self.vio_shift_spinbox.valueChanged.connect(self.vio_shift_update)
        self.gt_shift_spinbox.valueChanged.connect(self.gt_shift_update)

        self.prespect_plot.setBackground(0.3)
        self.two_d_plot.setBackground(0.3)

        self.path_color_r_spinbox.valueChanged.connect(self.color_update_R)
        self.path_color_g_spinbox.valueChanged.connect(self.color_update_G)
        self.path_color_b_spinbox.valueChanged.connect(self.color_update_B)


    def color_update_R(self):
        self.csv_dict[self.current_item].color[0] = self.path_color_r_spinbox.value()

    def color_update_G(self):
        self.csv_dict[self.current_item].color[1] = self.path_color_g_spinbox.value()

    def color_update_B(self):
        self.csv_dict[self.current_item].color[2] = self.path_color_b_spinbox.value()

    def animate(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._looper)
        self.timer.start(20)
        #self.start()

    def vio_shift_update(self, value):

        if (value - self.vio_shift_last) > 0:
            self.csv_dict[self.current_item].shift += 100000000
        else:
            self.csv_dict[self.current_item].shift -= 100000000

        self.vio_shift_last = value
        self.csv_dict[self.current_item].value_changed = True
        return
    
    def gt_shift_update(self, value):
        
        if (value - self.gt_shift_last) > 0:
            if self.gt_file != '' : 
                self.csv_dict[self.gt_file].shift += 100000000
        else:
            if self.gt_file != '' : 
                self.csv_dict[self.gt_file].shift -= 100000000

        self.gt_shift_last = value
        return

    def update_gt_timestamp(self, timestamp):

        if type(timestamp) not in [ float , int ]:
            print('type error')
            return
        
        if self.gt_file == '':
            print('no gt file')
            return 
        
        if not self.csv_dict[self.gt_file].has_timestamp :
            print('make time')

            self.csv_dict[self.gt_file].create_timestamp_via_Hz(timestamp)
            self.csv_dict[self.gt_file].has_timestamp = True
            self.csv_dict[self.gt_file].value_changed = True
            

    def open_gt_file(self):
        
        self.timer.stop()
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

        self.VIO_file.setEnabled(True)
        self.auto_align_button.setEnabled(True)
        self.timer.start()

    def open_vio_file(self):
        
        self.timer.stop()
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

        self.timer.start()

    def remove_item(self, item):

        target = item.text()
        self.csv_dict.pop(target)
        self.listWidget_CsvList.takeItem(self.listWidget_CsvList.currentRow())

        if target == self.gt_file:
            self.VIO_file.setEnabled(False)
            self.auto_align_button.setEnabled(False)

    def update_frame(self, item):
        
        target = item.text()
        self.current_item = target
        csv_class = self.csv_dict[target]
        gt_class = self.csv_dict[self.gt_file]

        self.Zangle_spinbox.setValue(csv_class.z_rotate)
        self.Xangle_spinbox.setValue(csv_class.y_rotate)
        self.Yangle_spinbox.setValue(csv_class.x_rotate)

        self.xscale_spinbox.setValue(csv_class.x_scale)
        self.yscale_spinbox.setValue(csv_class.y_scale)

        self.path_color_r_spinbox.setValue(csv_class.color[0])
        self.path_color_g_spinbox.setValue(csv_class.color[1])
        self.path_color_b_spinbox.setValue(csv_class.color[2])

        self.vio_shift_spinbox.setValue(int(csv_class.shift/100000000))
        self.vio_shift_spinbox.setValue(int(gt_class.shift/100000000))

        if not math.isnan(csv_class.start_time) :
            self.start_time_spinbox.setValue(int(csv_class.start_time/1000000000000))

        if not math.isnan(csv_class.end_time) :  
            self.end_time_spinbox.setValue(int(csv_class.end_time/1000000000000))

    def update_x_scale(self, value):

        if self.current_item == '':
                return 
        
        self.csv_dict[self.current_item].x_scale = value  # Mapping slider value to a reasonable range
        self.csv_dict[self.current_item].value_changed = True

    def update_y_scale(self, value):

        if self.current_item == '':
                return
        
        self.csv_dict[self.current_item].y_scale = value
        self.csv_dict[self.current_item].value_changed = True

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
        self.csv_dict[self.current_item].value_changed = True

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

        self.csv_dict[self.current_item].value_changed = True

    def job(self):

        [ csv.ProcessData() for name, csv in self.csv_dict.items()]

    def process_data(self):

        self.timer_process = QtCore.QTimer()
        self.timer_process.timeout.connect(self.job)
        self.timer_process.start(100)
        

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

            csv.SaveModifyCsv(self.start_from_index, self.end_to_index)

    def _looper(self):
        
        self.prespect_plot.clear()
        self.two_d_plot.clear()
        self.three_d_plot.clear()
        
        self.three_d_plot.addItem(gl.GLGridItem())

        for name, csv in self.csv_dict.copy().items():
 
            if type(csv) != Csv_Manager:
                continue

            plot_cache = csv.cache_data.copy()

            spl = gl.GLLinePlotItem(pos = list(plot_cache[['px','py','pz']].itertuples(index=False, name=None)), 
                                       color = (
                                                       csv.color[0], 
                                                       csv.color[1], 
                                                       csv.color[2],
                                                       .5),
                                        mode = 'line_strip', width = 2)
            
            self.three_d_plot.addItem(spl)
            self.prespect_plot.plot(plot_cache['px'], 
                                    plot_cache['py'],
                                    pen = (
                                            csv.color[0]*255, 
                                            csv.color[1]*255, 
                                            csv.color[2]*255,
                                            ),
                                    width=2)
            
            self.two_d_plot.plot(plot_cache['timestamp'].astype(float), 
                                 plot_cache['px'],
                                 pen = (
                                            csv.color[0]*255, 
                                            csv.color[1]*255, 
                                            csv.color[2]*255,
                                            ),
                                width=2
                                )


    def start(self):
        QtWidgets.QApplication.instance().exec()

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
window.animate()
window.process_data()
app.exec()