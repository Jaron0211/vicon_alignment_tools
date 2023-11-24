import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

# 讀取 CSV 檔案，指定列索引
file_path = 'dgvins_vio.csv'
data = pd.read_csv(file_path, header=None, index_col=False, names=['timestamp', 'pos_x', 'pos_y', 'pos_z', 'q_w', 'q_x', 'q_y', 'q_z'])

# 取得資料
timestamps = data['timestamp']
positions = data[['pos_x', 'pos_y', 'pos_z']]
quaternions = data[['q_w', 'q_x', 'q_y', 'q_z']]

# 設定 3D 圖形
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

# 繪製軌跡，設定顏色
norm = (data.index - data.index.min()) / (data.index.max() - data.index.min())  # Normalize to [0, 1]
colors = plt.cm.hsv(norm)[:, :3]  # Use 'viridis' colormap for RGB values

# Set up 3D plot
ax.set_title(file_path)
ax.plot(data['pos_x'], data['pos_y'], data['pos_z'], linestyle='-', color='gray', alpha=0.3)
ax.scatter3D(positions['pos_x'], positions['pos_y'], positions['pos_z'], c=colors, label='Trajectory',marker='o',s=2)

# 繪製旋轉箭頭
for i in range(0, len(positions), int(len(positions)*0.008)):
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
file_path = 'data_aligned.csv'
data = pd.read_csv(file_path, header=None, index_col=False, names=['timestamp', 'pos_x', 'pos_y', 'pos_z', 'q_w', 'q_x', 'q_y', 'q_z'])

# 取得資料
timestamps = data['timestamp']
positions = data[['pos_x', 'pos_y', 'pos_z']]
quaternions = data[['q_w', 'q_x', 'q_y', 'q_z']]

# 設定 3D 圖形
bx = fig.add_subplot(122, projection='3d')

# Convert index to RGB values
norm = (data.index - data.index.min()) / (data.index.max() - data.index.min())  # Normalize to [0, 1]
colors = plt.cm.hsv(norm)[:, :3]  # Use 'viridis' colormap for RGB values

# Set up 3D plot
bx.set_title(file_path)
bx.scatter3D(positions['pos_x'], positions['pos_y'], positions['pos_z'], c=colors, label='Trajectory',marker='o',s=.5)


# 繪製旋轉箭頭
for i in range(0, len(positions), int(len(positions)*0.008)):
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
    bx.quiver(positions.iloc[i]['pos_x'], positions.iloc[i]['pos_y'], positions.iloc[i]['pos_z'],
              arrow_length * x_axis_rotated[0], arrow_length * x_axis_rotated[1], arrow_length * x_axis_rotated[2],
              color='red')
    bx.quiver(positions.iloc[i]['pos_x'], positions.iloc[i]['pos_y'], positions.iloc[i]['pos_z'],
              arrow_length * y_axis_rotated[0], arrow_length * y_axis_rotated[1], arrow_length * y_axis_rotated[2],
              color='green')
    bx.quiver(positions.iloc[i]['pos_x'], positions.iloc[i]['pos_y'], positions.iloc[i]['pos_z'],
              arrow_length * z_axis_rotated[0], arrow_length * z_axis_rotated[1], arrow_length * z_axis_rotated[2],
              color='blue')

# 設定坐標軸標籤
bx.set_xlabel('X')
bx.set_ylabel('Y')
bx.set_zlabel('Z')

# 設定圖例
bx.legend()

# 顯示圖形
for angle in range(0, 360*4 + 1):
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
    
    ax.view_init(30, azim)
    bx.view_init(30, azim)
    #plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))

    plt.draw()
    plt.pause(.0005)

plt.close()
