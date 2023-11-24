import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 讀取 CSV 檔案，指定列索引
file_path = 'dgvins_vio.csv'
data = pd.read_csv(file_path, header=None, index_col=False, names=['timestamp', 'pos_x', 'pos_y', 'pos_z', 'q_w', 'q_x', 'q_y', 'q_z'])
print(data)

# 取得資料
timestamps = data['timestamp']
positions = data[['pos_x', 'pos_y', 'pos_z']]
quaternions = data[['q_w', 'q_x', 'q_y', 'q_z']]

# 設定 3D 圖形
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

# 繪製軌跡
ax.set_title(file_path)
ax.plot(positions['pos_x'], positions['pos_y'], positions['pos_z'], label='Trajectory')

# 繪製旋轉箭頭
for i in range(0, len(positions), int(len(positions)*0.005)):
    rotation_matrix = np.array([
        [1 - 2*(quaternions['q_y'][i]**2 + quaternions['q_z'][i]**2), 2*(quaternions['q_x'][i]*quaternions['q_y'][i] - quaternions['q_w'][i]*quaternions['q_z'][i]), 2*(quaternions['q_x'][i]*quaternions['q_z'][i] + quaternions['q_w'][i]*quaternions['q_y'][i])],
        [2*(quaternions['q_x'][i]*quaternions['q_y'][i] + quaternions['q_w'][i]*quaternions['q_z'][i]), 1 - 2*(quaternions['q_x'][i]**2 + quaternions['q_z'][i]**2), 2*(quaternions['q_y'][i]*quaternions['q_z'][i] - quaternions['q_w'][i]*quaternions['q_x'][i])],
        [2*(quaternions['q_x'][i]*quaternions['q_z'][i] - quaternions['q_w'][i]*quaternions['q_y'][i]), 2*(quaternions['q_y'][i]*quaternions['q_z'][i] + quaternions['q_w'][i]*quaternions['q_x'][i]), 1 - 2*(quaternions['q_x'][i]**2 + quaternions['q_y'][i]**2)]
    ])

    # 取得旋轉後的坐標軸
    x_axis_rotated = rotation_matrix @ np.array([1, 0, 0])
    y_axis_rotated = rotation_matrix @ np.array([0, 1, 0])
    z_axis_rotated = rotation_matrix @ np.array([0, 0, 1])

    # 繪製箭頭
    arrow_length = 0.1
    ax.quiver(positions.iloc[i]['pos_x'], positions.iloc[i]['pos_y'], positions.iloc[i]['pos_z'],
              arrow_length * x_axis_rotated[0], arrow_length * x_axis_rotated[1], arrow_length * x_axis_rotated[2],
              color='red')
    ax.quiver(positions.iloc[i]['pos_x'], positions.iloc[i]['pos_y'], positions.iloc[i]['pos_z'],
              arrow_length * y_axis_rotated[0], arrow_length * y_axis_rotated[1], arrow_length * y_axis_rotated[2],
              color='green')
    ax.quiver(positions.iloc[i]['pos_x'], positions.iloc[i]['pos_y'], positions.iloc[i]['pos_z'],
              arrow_length * z_axis_rotated[0], arrow_length * z_axis_rotated[1], arrow_length * z_axis_rotated[2],
              color='blue')

# 設定坐標軸標籤
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 設定圖例
ax.legend()

##next
file_path = 'data_org.csv'
data = pd.read_csv(file_path, header=None, index_col=False, names=['timestamp', 'pos_x', 'pos_y', 'pos_z', 'q_w', 'q_x', 'q_y', 'q_z'])
print(data)

# 取得資料
timestamps = data['timestamp']
positions = data[['pos_x', 'pos_y', 'pos_z']]
quaternions = data[['q_w', 'q_x', 'q_y', 'q_z']]

# 設定 3D 圖形

ax = fig.add_subplot(122, projection='3d')

# 繪製軌跡
ax.set_title(file_path)
ax.plot(positions['pos_x'], positions['pos_y'], positions['pos_z'], label='Trajectory')

# 繪製旋轉箭頭
for i in range(0, len(positions), int(len(positions)*0.005)):
    rotation_matrix = np.array([
        [1 - 2*(quaternions['q_y'][i]**2 + quaternions['q_z'][i]**2), 2*(quaternions['q_x'][i]*quaternions['q_y'][i] - quaternions['q_w'][i]*quaternions['q_z'][i]), 2*(quaternions['q_x'][i]*quaternions['q_z'][i] + quaternions['q_w'][i]*quaternions['q_y'][i])],
        [2*(quaternions['q_x'][i]*quaternions['q_y'][i] + quaternions['q_w'][i]*quaternions['q_z'][i]), 1 - 2*(quaternions['q_x'][i]**2 + quaternions['q_z'][i]**2), 2*(quaternions['q_y'][i]*quaternions['q_z'][i] - quaternions['q_w'][i]*quaternions['q_x'][i])],
        [2*(quaternions['q_x'][i]*quaternions['q_z'][i] - quaternions['q_w'][i]*quaternions['q_y'][i]), 2*(quaternions['q_y'][i]*quaternions['q_z'][i] + quaternions['q_w'][i]*quaternions['q_x'][i]), 1 - 2*(quaternions['q_x'][i]**2 + quaternions['q_y'][i]**2)]
    ])

    # 取得旋轉後的坐標軸
    x_axis_rotated = rotation_matrix @ np.array([1, 0, 0])
    y_axis_rotated = rotation_matrix @ np.array([0, 1, 0])
    z_axis_rotated = rotation_matrix @ np.array([0, 0, 1])

    # 繪製箭頭
    arrow_length = 0.1
    ax.quiver(positions.iloc[i]['pos_x'], positions.iloc[i]['pos_y'], positions.iloc[i]['pos_z'],
              arrow_length * x_axis_rotated[0], arrow_length * x_axis_rotated[1], arrow_length * x_axis_rotated[2],
              color='red')
    ax.quiver(positions.iloc[i]['pos_x'], positions.iloc[i]['pos_y'], positions.iloc[i]['pos_z'],
              arrow_length * y_axis_rotated[0], arrow_length * y_axis_rotated[1], arrow_length * y_axis_rotated[2],
              color='green')
    ax.quiver(positions.iloc[i]['pos_x'], positions.iloc[i]['pos_y'], positions.iloc[i]['pos_z'],
              arrow_length * z_axis_rotated[0], arrow_length * z_axis_rotated[1], arrow_length * z_axis_rotated[2],
              color='blue')

# 設定坐標軸標籤
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 設定圖例
ax.legend()

# 顯示圖形
plt.show()
