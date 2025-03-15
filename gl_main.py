from csv_manager import Csv_Manager

import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl

from scipy.spatial.transform import Rotation as R

import pyqtgraph as pg
import math, threading
import time
import pandas as pd
import numpy as np
import os
import aligment_toolbox

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
        self.Rot_pre = np.array([[1,0,0],[0,1,0],[0,0,1]])

        self.start_from_index = -1
        self.end_to_index = -1

        self.threading_lock = False

        #ui item
        self.VIO_file.setEnabled(False)
        self.auto_align_button.setEnabled(False)

        self.GT_file.clicked.connect(self.toggle_gt_file)
        self.VIO_file.clicked.connect(self.open_vio_file)
        self.save_result.clicked.connect(self.save_alignment)

        self.yscale_spinbox.valueChanged.connect(self.update_y_scale)
        self.xscale_spinbox.valueChanged.connect(self.update_x_scale)

        self.listWidget_CsvList.itemClicked.connect(self.update_frame)
        self.listWidget_CsvList.itemDoubleClicked.connect(self.remove_item)

        self.dial_vio_shift.valueChanged.connect(self.vio_shift_update)

        self.prespect_plot.setBackground(0.3)
        self.two_d_plot.setBackground(0.3)

        def changestate():
            if self.current_item != '':
                self.csv_dict[self.current_item].value_changed = True


        self.Zangle_spinbox.valueChanged.connect(lambda: changestate())
        self.Yangle_spinbox.valueChanged.connect(lambda: changestate())
        self.Xangle_spinbox.valueChanged.connect(lambda: changestate())
        self.trans_z_spinbox.valueChanged.connect(lambda: changestate())
        self.trans_y_spinbox.valueChanged.connect(lambda: changestate())
        self.trans_x_spinbox.valueChanged.connect(lambda: changestate())

        self.path_color_r_spinbox.valueChanged.connect(self.color_update_r)
        self.path_color_g_spinbox.valueChanged.connect(self.color_update_g)
        self.path_color_b_spinbox.valueChanged.connect(self.color_update_b)

        self.auto_align_button.clicked.connect(self.auto_alignment)

        self.TR_updater = QtCore.QTimer()
        self.TR_updater.timeout.connect(lambda: self.update_path())

        self.looper_timer = QtCore.QTimer()
        self.looper_timer.timeout.connect(lambda: self._looper())
        
        self.process_timer = QtCore.QTimer()
        self.process_timer.timeout.connect(lambda: self.csv_process())
        
        self.alignment_worker = None

        self.init_timers()

    def init_timers(self):
        self.looper_timer.start(10)
        self.process_timer.start(10)
        self.TR_updater.start(10)

    def closeEvent(self, event):
        self.looper_timer.stop()
        self.process_timer.stop()
        self.TR_updater.stop()
        if hasattr(self, 'icp_thread') and self.icp_thread.isActive():
            self.icp_thread.stop()
        if self.alignment_worker is not None:
            self.alignment_worker.terminate()
        event.accept()

    def timer_start(self):
        self.init_timers()

    def color_update_r(self):
        if self.current_item == '':
            return
        self.csv_dict[self.current_item].color[0] = self.path_color_r_spinbox.value()
    def color_update_g(self):
        if self.current_item == '':
            return
        self.csv_dict[self.current_item].color[1] = self.path_color_g_spinbox.value()
    def color_update_b(self):
        if self.current_item == '':
            return    
        self.csv_dict[self.current_item].color[2] = self.path_color_b_spinbox.value()

    def auto_alignment(self):
        # If currently running, stop the alignment
        if hasattr(self, 'icp_thread') and self.icp_thread.isActive():
            self.icp_thread.stop()
            self.auto_align_button.setText("Auto Alignment")
            self.auto_align_button.setEnabled(True)
            return

        if self.gt_file == '' or self.current_item == '':
            return

        # Reset transformations
        self.Rot_pre = np.eye(3)
        self.total_rot = np.eye(3)
        self.total_trans = np.zeros((3,))

        # Change button text to indicate stopping is possible
        self.auto_align_button.setText("Stop Alignment")

        SourceCache = self.csv_dict[self.gt_file].cache_data
        TargetCache = self.csv_dict[self.current_item].cache_data

        if self.start_from_index > 0 and self.end_to_index > 0:
            TargetCache = TargetCache[TargetCache['timestamp'].between(self.start_from_index, self.end_to_index, inclusive='both')]

        self.total_rot = np.array([-float(self.csv_dict[self.current_item].z_rotate), 
                                   -float(self.csv_dict[self.current_item].y_rotate), 
                                   -float(self.csv_dict[self.current_item].x_rotate)])
        
        self.total_rot = R.from_euler('ZYX', self.total_rot, True).as_matrix()

        self.total_trans = np.array([[float(self.csv_dict[self.current_item].x_transition), 
                                   float(self.csv_dict[self.current_item].y_transition), 
                                   float(self.csv_dict[self.current_item].z_transition)]])
        
        self.icp_thread = QtCore.QTimer()
        def job():
            Rot, Trans = aligment_toolbox.ICP(SourceCache, 
                                            TargetCache,
                                            SampleNum = 200)
            
            rotation_xyz = R.from_matrix(Rot.T)

            rotated_coordinates_xyz = np.column_stack((TargetCache['px'], TargetCache['py'], TargetCache['pz']))
            rotated_coordinates_xyz = rotation_xyz.apply(rotated_coordinates_xyz)

            TargetCache[['px','py','pz']] = rotated_coordinates_xyz
            TargetCache[['px','py','pz']] -= Trans

            self.total_rot = rotation_xyz.apply(self.total_rot)
            cost = self.Rot_pre.T @ Rot - np.identity(3)
            result = R.from_matrix(self.total_rot.T)

            TargetCacheCentroid = np.asarray(self.csv_dict[self.current_item].path_data[['px','py','pz']].mean())
            SourceCacheCentroid = np.asarray(TargetCache[['px','py','pz']].mean())

            self.total_trans = self.total_rot.T @ TargetCacheCentroid - SourceCacheCentroid

            result = R.from_matrix(self.total_rot)
            result = result.as_euler('zyx',True)

            self.csv_dict[self.current_item].z_rotate = -result[0]
            self.csv_dict[self.current_item].y_rotate = -result[1]
            self.csv_dict[self.current_item].x_rotate = -result[2]

            self.csv_dict[self.current_item].x_transition = self.total_trans[0]
            self.csv_dict[self.current_item].y_transition = self.total_trans[1]
            self.csv_dict[self.current_item].z_transition = self.total_trans[2]

            self.Zangle_spinbox.setValue(self.csv_dict[self.current_item].z_rotate)
            self.Yangle_spinbox.setValue(self.csv_dict[self.current_item].y_rotate)
            self.Xangle_spinbox.setValue(self.csv_dict[self.current_item].x_rotate)

            self.trans_x_spinbox.setValue(self.total_trans[0])
            self.trans_y_spinbox.setValue(self.total_trans[1])
            self.trans_z_spinbox.setValue(self.total_trans[2])

            self.Rot_pre = Rot
            
            if ((abs(cost) < 10**-6).all()):
                # Create worker thread for alignment
                class AlignmentWorker(QtCore.QThread):
                    finished = QtCore.pyqtSignal(tuple)
                    
                    def __init__(self, source, target):
                        super().__init__()
                        self.source = source
                        self.target = target  # Fix the initialization of target
                        
                    def run(self):
                        result = aligment_toolbox.AlignmentPath(self.source, self.target)
                        self.finished.emit(result)

                # Create and start worker thread
                # Stop ICP thread before starting alignment
                self.icp_thread.stop()
                self.auto_align_button.setText("Auto Alignment") 

                # Create and run alignment worker once
                self.alignment_worker = AlignmentWorker(SourceCache, TargetCache)
                self.alignment_worker.finished.connect(self.finish_alignment)
                self.alignment_worker.start()
                return
                
        self.icp_thread.timeout.connect(lambda: job())
        self.icp_thread.start()

    def finish_alignment(self, result):
        SourceCache_1, _, BestTimeShift = result
        print(f"Alignment result: {SourceCache_1.head()}")
        print(f"Best time shift: {BestTimeShift}")
        self.csv_dict[self.gt_file].cache_data['timestamp'] = SourceCache_1['timestamp']
        self.csv_dict[self.current_item].shift = BestTimeShift
        self.auto_align_button.setText("Auto Alignment")
        self.auto_align_button.setEnabled(True)
        self.icp_thread.stop()
        self.alignment_worker = None

    def vio_shift_update(self, value):

        if (value - self.vio_shift_last) > 0:
            self.csv_dict[self.current_item].shift += 100000000
        else:
            self.csv_dict[self.current_item].shift -= 100000000

        self.vio_shift_last = value
        self.csv_dict[self.current_item].value_changed = True

        if not math.isnan(self.csv_dict[self.current_item].start_time) :
            self.start_time_spinbox.setValue(int(self.csv_dict[self.current_item].start_time/100000000))

        if not math.isnan(self.csv_dict[self.current_item].end_time) :  
            self.end_time_spinbox.setValue(int(self.csv_dict[self.current_item].end_time/100000000))

        print(self.csv_dict[self.current_item].shift)
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

    def toggle_gt_file(self):
        if self.gt_file:
            self.remove_gt_file()
        else:
            self.open_gt_file()

    def open_gt_file(self):
        self.threading_lock = True
        gt_file, filetype = QtWidgets.QFileDialog.getOpenFileName(self,  
                                    "Choose the GT path file",  
                                    self.cwd, # 起始路径 
                                    "EuRoC Files (*.csv);;TUM Files (*.tum);;All Files (*)")
        
        _name = gt_file.split('/')[-1]
        if gt_file == "" or (_name in self.csv_dict):
            self.threading_lock = False
            return
        
        self.csv_dict[_name] = Csv_Manager(gt_file)

        self.listWidget_CsvList.addItem(_name)
        self.listWidget_CsvList.findItems(_name, Qt.MatchExactly)[0].setForeground(Qt.red)
        self.gt_file = _name

        self.VIO_file.setEnabled(True)
        self.auto_align_button.setEnabled(True)
        self.GT_file.setText("Delete GT File")
        self.threading_lock = False

    def remove_gt_file(self):
        if self.gt_file:
            self.threading_lock = True
            self.csv_dict.pop(self.gt_file)
            items = self.listWidget_CsvList.findItems(self.gt_file, Qt.MatchExactly)
            if items:
                self.listWidget_CsvList.takeItem(self.listWidget_CsvList.row(items[0]))
            self.gt_file = ''
            self.VIO_file.setEnabled(False)
            self.auto_align_button.setEnabled(False)
            self.GT_file.setText("Open GT File")
            self.threading_lock = False

    def open_vio_file(self):
        
        self.threading_lock = True
        vio_file, filetype = QtWidgets.QFileDialog.getOpenFileName(self,  
                                    "Choose the VIO path file",  
                                    self.cwd, # 起始路径 
                                    "EuRoC Files (*.csv);;TUM Files (*.tum);;All Files (*)") 

        _name = vio_file.split('/')[-1]

        if vio_file == "" or (_name in self.csv_dict):
            return
        
        self.csv_dict[_name] = Csv_Manager(vio_file)
        self.listWidget_CsvList.addItem(_name)
        self.listWidget_CsvList.findItems(_name, Qt.MatchExactly)[0].setForeground(Qt.gray)

        self.threading_lock = False

    def remove_item(self, item):
        self.threading_lock = True

        target = item.text()

        if target == self.gt_file:
            self.remove_gt_file()
        if target == self.current_item:
            self.current_item = ''

        if target in self.csv_dict:
            self.csv_dict.pop(target)
            self.listWidget_CsvList.takeItem(self.listWidget_CsvList.currentRow())

        self.threading_lock = False

    def update_frame(self, item):
        
        target = item.text()
        
        self.current_item = target
        csv_class = self.csv_dict[target]

        self.Zangle_spinbox.setValue(csv_class.z_rotate)
        self.Yangle_spinbox.setValue(csv_class.y_rotate)
        self.Xangle_spinbox.setValue(csv_class.x_rotate)

        self.trans_x_spinbox.setValue(csv_class.x_transition)
        self.trans_y_spinbox.setValue(csv_class.y_transition)
        self.trans_z_spinbox.setValue(csv_class.z_transition)

        self.xscale_spinbox.setValue(csv_class.x_scale)
        self.yscale_spinbox.setValue(csv_class.y_scale)

        self.path_color_r_spinbox.setValue(csv_class.color[0])
        self.path_color_g_spinbox.setValue(csv_class.color[1])
        self.path_color_b_spinbox.setValue(csv_class.color[2])

        if not math.isnan(csv_class.start_time) :
            self.start_time_spinbox.setValue(int(csv_class.start_time/100000000))
        else:
            self.start_time_spinbox.setValue(0)

        if not math.isnan(csv_class.end_time) :  
            self.end_time_spinbox.setValue(int(csv_class.end_time/100000000))
        else:
            self.end_time_spinbox.setValue(0)

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

    def update_start_from_and_end_to(self):
        
        if self.gt_file == '' : return
        x_min ,x_max = self.csv_dict[self.gt_file].cache_data['timestamp'].min(), self.csv_dict[self.gt_file].cache_data['timestamp'].max()

        self.start_from_index =  x_min + self.Start_from.value() /100 * (x_max - x_min)
        self.end_to_index = x_max - self.End_to.value() /100 * (x_max - x_min)

        # if self.start_from_index > self.end_to_index:
        #     self.start_from_index = self.end_to_index

    def update_path(self):

        if self.threading_lock:
            return

        self.update_transition()
        self.update_rotation()

    def update_transition(self):

        if self.current_item == '':
                return
        
        self.csv_dict[self.current_item].z_transition = self.trans_z_spinbox.value()  
        self.csv_dict[self.current_item].y_transition = self.trans_y_spinbox.value()
        self.csv_dict[self.current_item].x_transition = self.trans_x_spinbox.value()

    def update_rotation(self):

        if self.current_item == '':
                return
        
        self.csv_dict[self.current_item].z_rotate = self.Zangle_spinbox.value()
        self.csv_dict[self.current_item].y_rotate = self.Yangle_spinbox.value()
        self.csv_dict[self.current_item].x_rotate = self.Xangle_spinbox.value()


    def csv_process(self):
        
        if self.threading_lock:
            return
        
        #[ threading.Thread(target = csv.ProcessData, name = name).start() for name, csv in self.csv_dict.items()]        
        #[ csv.ProcessData() for name, csv in self.csv_dict.items()]        

    def save_alignment(self):

        for name, csv in self.csv_dict.items():
            if type(csv) != Csv_Manager:
                continue

            if self.start_from_index > 0 and self.end_to_index > 0:
                csv.SaveModifyCsv(self.start_from_index, self.end_to_index)

    def _looper(self):
        if self.threading_lock:
            return
        
        # Clear plots once
        self.prespect_plot.clear()
        self.two_d_plot.clear()
        self.three_d_plot.clear()
        
        # Add static items
        self.three_d_plot.addItem(gl.GLAxisItem(size=QVector3D(1.0,1.0,1.0),glOptions='opaque'))
        self.three_d_plot.addItem(gl.GLGridItem())

        self.update_start_from_and_end_to()

        # Process each CSV
        for name, csv in self.csv_dict.items():
            if not isinstance(csv, Csv_Manager):
                continue

            plot_cache = csv.cache_data
            color = csv.color
            color_rgb = (color[0]*255, color[1]*255, color[2]*255)
            color_rgb_dim = (color[0]*127.5, color[1]*127.5, color[2]*127.5)
            color_3d = (color[0], color[1], color[2], 0.5)
            color_3d_dim = (color[0], color[1], color[2], 0.2)

            # Apply timestamp shift
            plot_timestamps = plot_cache['timestamp'].values - csv.shift

            if self.start_from_index > 0 and self.end_to_index > 0:
                # Use boolean indexing instead of multiple filters
                mask_in = (plot_timestamps >= self.start_from_index) & (plot_timestamps <= self.end_to_index)
                plot_cache_in = plot_cache[mask_in]
                timestamps_in = plot_timestamps[mask_in]
                plot_cache_out_down = plot_cache[plot_timestamps < self.start_from_index]
                timestamps_out_down = plot_timestamps[plot_timestamps < self.start_from_index]
                plot_cache_out_up = plot_cache[plot_timestamps > self.end_to_index]
                timestamps_out_up = plot_timestamps[plot_timestamps > self.end_to_index]

                # Plot 2D views
                for cache, times, color in [(plot_cache_in, timestamps_in, color_rgb), 
                                          (plot_cache_out_down, timestamps_out_down, color_rgb_dim),
                                          (plot_cache_out_up, timestamps_out_up, color_rgb_dim)]:
                    if len(cache) > 0:
                        self.two_d_plot.plot(times, 
                                           cache['pz'].values,
                                           pen=color, width=2)
                        self.prespect_plot.plot(cache['px'].values,
                                              cache['py'].values,
                                              pen=color, width=2)

                # Plot 3D views
                for cache, color in [(plot_cache_in, color_3d),
                                   (plot_cache_out_down, color_3d_dim),
                                   (plot_cache_out_up, color_3d_dim)]:
                    if len(cache) > 0:
                        pos = cache[['px','py','pz']].values
                        spl = gl.GLLinePlotItem(pos=pos, color=color,
                                              mode='line_strip', width=1,
                                              glOptions='opaque')
                        self.three_d_plot.addItem(spl)

                # Add range indicator lines
                if self.start_from_index > 0 and self.end_to_index > 0:
                    self.two_d_plot.addItem(
                        pg.InfiniteLine(self.start_from_index, angle=90, 
                                      pen=(255,0,0), movable=False),
                        ignoreBounds=True)
                    self.two_d_plot.addItem(
                        pg.InfiniteLine(self.end_to_index, angle=90,
                                      pen=(0,0,255), movable=False),
                        ignoreBounds=True)
            else:
                # Plot full range
                self.prespect_plot.plot(plot_cache['px'].values,
                                      plot_cache['py'].values,
                                      pen=color_rgb, width=2)
                self.two_d_plot.plot(plot_timestamps,
                                   plot_cache['pz'].values,
                                   pen=color_rgb, width=2)
                pos = plot_cache[['px','py','pz']].values
                spl = gl.GLLinePlotItem(pos=pos, color=color_3d,
                                      mode='line_strip', width=1,
                                      glOptions='opaque')
                self.three_d_plot.addItem(spl)

    def start(self):
        QtWidgets.QApplication.instance().exec()

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
window.timer_start()
app.exec()