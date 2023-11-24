import pandas as pd
import numpy as np
from tslearn.metrics import dtw_path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

# 讀取A資料
A_data = pd.read_csv('dgvins_vio.csv', header=None, names=['timestamp', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10'])

# 讀取B資料，並加上虛構的時間標記
B_data = pd.read_csv('data_aligned.csv', header=None, names=['aligned_timestamp', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8'])
B_data_origin = B_data.copy()
# # 將 x, y, z 座標進行逆時針旋轉 90 度
rotated_coordinates = np.column_stack((B_data['col1'], B_data['col2'], B_data['col3']))
rotation = R.from_euler('z', -90, degrees=True)
rotated_coordinates = rotation.apply(rotated_coordinates)

# 更新 B_data 的 col2, col3, col4 列為旋轉後的值
B_data[['col1', 'col2', 'col3']] = rotated_coordinates
B_data[['aligned_timestamp', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8']].to_csv('data_aligned_rotated.csv', index=False, header=None, float_format='%.9f')

# 3D 繪圖，調整點的大小
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(B_data['col1'], B_data['col2'], B_data['col3'], s=1, label='Rotated Data')  # 調整 s 參數
ax.scatter(A_data['col1'], A_data['col2'], A_data['col3'], s=1, label='dgvins Data')  # 調整 s 參數
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
