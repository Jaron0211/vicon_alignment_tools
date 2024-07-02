import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

class Csv_Manager():

    def __init__(self, csv_file):

        self.path_data = pd.read_csv(csv_file, header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])

        self.name = csv_file.split('/')[-1]
        self.len = len(self.path_data['timestamp'])
        
        self.shift = 0
        self.cache_data = self.path_data.copy()

        self.interpolation_timestamp = self.path_data['timestamp'].copy()

        self.has_timestamp = True
        if (self.path_data['timestamp'].isna().any()):
            self.has_timestamp = False

        self.pos=[]

        self.start_time = self.path_data['timestamp'].iloc[0]
        self.end_time   = self.path_data['timestamp'].iloc[-1]

        self.y_scale = 1.0
        self.x_scale = 1.0

        self.y_scale_pre = self.y_scale
        self.x_scale_pre = self.x_scale

        self.z_rotate = 0
        self.y_rotate = 0
        self.x_rotate = 0

        self.z_transition = 0
        self.y_transition = 0
        self.x_transition = 0

        self.color=[1.0,1.0,1.0]

        self.value_changed = False

        self.delta_time = self.path_data['timestamp'].diff()
    
    def SaveModifyCsv(self, start_time, end_time):

        self.cache_data[self.cache_data['timestamp'].between(start_time, end_time, inclusive="both")]
        self.cache_data['timestamp'] = self.cache_data['timestamp'].astype(float)  
        self.cache_data.to_csv("Aligned_{}.csv".format(self.name) ,index=False, header=None, float_format='%.9f')
        
    def ProcessData(self):

        if not self.value_changed:
            return
        
        rotated_coordinates_xyz = np.column_stack((self.path_data['px'], self.path_data['py'], self.path_data['pz']))
        rotation_xyz = R.from_euler('ZYX', [self.z_rotate,self.y_rotate,self.x_rotate], degrees=True)
        rotated_coordinates_xyz = rotation_xyz.apply(rotated_coordinates_xyz)
        
        self.cache_data[['px','py','pz']] = rotated_coordinates_xyz * self.y_scale

        _starttime = self.path_data['timestamp'][0]

        self.interpolation_timestamp[0] = _starttime + self.shift

        def job():
            def iterrator(i):
                self.interpolation_timestamp[i] = self.interpolation_timestamp[i-1] + self.delta_time[i] * self.x_scale 
                
            if len(self.path_data['timestamp']) == len(self.path_data.index):
               [iterrator(i) for i in range(1, len(self.path_data['timestamp']),1)]

            self.cache_data['timestamp'] = self.interpolation_timestamp 

        job()
        self.value_changed = False


    def create_timestamp_via_Hz(self, start_time, hz = 100):

        self.path_data['timestamp'] = [ (start_time + (i)/self.x_scale*10**7 ) for i in range(0, self.len, 1)]
        self.delta_time = self.path_data['timestamp'].diff()

        self.start_time = self.path_data['timestamp'].iloc[0]
        self.end_time   = self.path_data['timestamp'].iloc[-1]