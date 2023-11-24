import pandas as pd
import numpy as np
from tslearn.metrics import dtw_path
import matplotlib.pyplot as plt

# 讀取A資料
A_data = pd.read_csv('dgvins_vio.csv', header=None, names=['timestamp', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10'])
# timestamp, x,y,z, w,x,y,z, vx,vy,vz

# 讀取B資料，並加上虛構的時間標記
B_data = pd.read_csv('data_org.csv', header=None, names=['dummy', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8'])
# timestamp, x,y,z, x,y,z,w

# 假設B資料每筆資料之間的時間差為固定值，這裡假設是0.01秒
time_difference = 0.01
B_data['timestamp'] = np.arange(0, len(B_data) * time_difference, time_difference)[:-1]

# 對B資料進行內插，填充時間標記
B_data['timestamp'] = B_data['timestamp'].interpolate(method='linear')

# 將A和B資料的軌跡欄位取出
A_trajectory = np.array(A_data[['col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8']])
B_trajectory = np.array(B_data[['col2', 'col3', 'col4', 'col8', 'col5', 'col6', 'col7']])

# 進行 DTW 對齊
print('ok')
path, distance = dtw_path(A_trajectory, B_trajectory)
print('ok')

# 限制 path 的值在有效範圍內並轉換為整數
path = np.clip(path, 0, len(A_data) - 1).astype(int)

# 將 path 的長度固定為 len(B_data)
path = np.linspace(0, len(A_data) - 1, len(B_data)).astype(int)

# 將 B_data 的 aligned_timestamp 列設置為 A_data 的 timestamp 列的相應值
B_data['aligned_timestamp'] = A_data['timestamp'].values[path].ravel()

# 存儲對齊後的資料
B_data[['aligned_timestamp', 'col1', 'col2', 'col3', 'col7', 'col4', 'col5', 'col6']].to_csv('data_aligned.csv', index=False, header=None, float_format='%.9f')

# # 繪製原始軌跡和對齊後的軌跡
# plt.plot(A_data['timestamp'], A_trajectory, label='A_data')
# plt.plot(B_data['aligned_timestamp'], B_trajectory, label='B_data (aligned)')
# plt.legend()
# plt.show()
