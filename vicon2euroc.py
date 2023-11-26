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

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import axes3d

vio_data = pd.read_csv(vio_file, header=None, names=['timestamp', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz'])
gt_data  = pd.read_csv(gt_file , header=None, names=['timestamp', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'])

fig = plt.figure()
ax = fig.add_subplot(111)

# force finding, maybe change to the DTW method
# https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html
for i in range(0,1500,1):
    offset = i
    ax.clear()
    
    vio_compare_array = vio_data['px'][0:200]*y_scale

    gt_indexs = [i for i in range(offset,offset+(x_scale*200),x_scale)]
    gt_compare_array = gt_data['py'][gt_indexs]

    error_array = abs(vio_compare_array-gt_compare_array)
    error_sum = error_array.sum()
    
    ax.plot(range(0,len(vio_compare_array),1), vio_compare_array)
    ax.plot(range(0,len(gt_compare_array),1), gt_compare_array )

    if error_sum < best_error_sum:
        best_error_sum = error_sum
        best_offset = offset

    plt.draw()
    plt.pause(0.0005)

print('The best offset: ', best_offset)

# Show the aligned result
fig2 = plt.figure('alignment_result')
cx = fig2.add_subplot(111)

gt_start = 1322
vio_end = int((len(gt_data['py'])-gt_start)/x_scale)
print(vio_end)

best_offset = 1322
cx.plot(range(best_offset,best_offset+len(vio_data['px'][:vio_end])*x_scale,x_scale),vio_data['px'][:1224]*y_scale)
cx.plot(gt_data['py'][1322:])
plt.show()

vio_data_crop = vio_data[:1224]
gt_data_crop  = gt_data[1322:]

time_start = vio_data_crop['timestamp'][ 0]
time_end   = vio_data_crop['timestamp'][len(vio_data_crop['timestamp'])-1]

print(time_start, time_end)

time_spend = time_end - time_start
len_of_vio = len(gt_data_crop['timestamp'])
time_space = int(time_spend / len_of_vio)

gt_data_crop['timestamp'][0] = time_start
gt_data_crop['timestamp'][len(gt_data_crop['timestamp'])-1] = time_end

timing_array = [i for i in range(int(time_start), int(time_end) ,time_space)]
gt_data_crop['timestamp'] = timing_array[:len(gt_data_crop['timestamp'])]


vio_data_crop.to_csv('timed_dgvins_vio.csv',index=False, header=None, float_format='%.9f')
gt_data_crop.to_csv('timed_gt.csv',index=False, header=None, float_format='%.9f')