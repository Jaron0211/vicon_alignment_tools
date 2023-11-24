import pandas as pd
import numpy as np
from tslearn.metrics import dtw_path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

# 讀取A資料
A_data = pd.read_csv('dgvins_vio.csv', header=None, names=['timestamp', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10'])

# 讀取B資料，並加上虛構的時間標記
B_data = pd.read_csv('data_org.csv', header=None, names=['aligned_timestamp', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7'])
B_data_origin = B_data.copy()

# 將 x, y, z 座標進行逆時針旋轉 90 度
rotated_coordinates_xyz = np.column_stack((B_data['col1'], B_data['col2'], B_data['col3']))
rotation_xyz = R.from_euler('z', -90, degrees=True)
rotated_coordinates_xyz = rotation_xyz.apply(rotated_coordinates_xyz)

# 更新 B_data 的 col1, col2, col3 列為旋轉後的值
B_data[['col1', 'col2', 'col3']] = rotated_coordinates_xyz

# 將四元數 (col4, col5, col6, col7) 轉換為 Rotation 類別
quaternions = B_data[['col4', 'col5', 'col6', 'col7']].values
rotation_quaternions = R.from_quat(quaternions)

# # 將四元數進行旋轉
# rotation_z = R.from_euler('z', 90, degrees=True)
# rotated_quaternions = rotation_z * rotation_quaternions

# rotation_z = R.from_euler('y', -90, degrees=True)
# rotated_quaternions = rotation_z * rotated_quaternions

# # 更新 B_data 的 col4, col5, col6, col7 列為旋轉後的四元數值
# B_data[['col4', 'col5', 'col6', 'col7']] = rotated_quaternions.as_quat()

# 將 B_data 寫入 CSV 檔案
B_data[['aligned_timestamp', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7']].to_csv('data_aligned_rotated.csv', index=False, header=None, float_format='%.9f')

# 3D 繪圖，調整點的大小
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(B_data['col1'], B_data['col2'], B_data['col3'], s=1, label='Rotated Data')  # 調整 s 參數
ax.scatter(A_data['col1'], A_data['col2'], A_data['col3'], s=1, label='dgvins Data')  # 調整 s 參數
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# bx = fig.add_subplot(122)
# bx.plot(B_data['aligned_timestamp'],B_data['col4'],label='Rotated Data')
# bx.plot(A_data['timestamp'],A_data['col4'],label='dgvins Data')
# bx.legend()
plt.show()
