import pandas as pd

load_gt = 1

# 載入資料
if (load_gt):
    print('1')
    file_path = "data_aligned_rotated.csv"
    data = pd.read_csv(file_path, header=None, index_col=False, names=['timestamp', 'pos_x', 'pos_y', 'pos_z', 'q_w', 'q_x', 'q_y', 'q_z'])
else:
    file_path = "dgvins_vio.csv"
    data = pd.read_csv(file_path, header=None, index_col=False, names=['timestamp', 'pos_x', 'pos_y', 'pos_z', 'q_w', 'q_x', 'q_y', 'q_z','vx','vy','vz'])

# 取得初始狀態
initial_state = data.iloc[0]

# 計算相對姿態
data['relative_pos_x'] = data['pos_x'] - initial_state['pos_x']
data['relative_pos_y'] = data['pos_y'] - initial_state['pos_y']
data['relative_pos_z'] = data['pos_z'] - initial_state['pos_z']

# 這裡的四元數相減需使用四元數運算
# 計算相對四元數
relative_q_w = data['q_w'] * initial_state['q_w'] + data['q_x'] * initial_state['q_x'] + data['q_y'] * initial_state['q_y'] + data['q_z'] * initial_state['q_z']
relative_q_x = data['q_w'] * initial_state['q_x'] - initial_state['q_w'] * data['q_x'] + data['q_y'] * initial_state['q_z'] - initial_state['q_y'] * data['q_z']
relative_q_y = data['q_w'] * initial_state['q_y'] - initial_state['q_w'] * data['q_y'] + initial_state['q_x'] * data['q_z'] - data['q_x'] * initial_state['q_z']
relative_q_z = data['q_w'] * initial_state['q_z'] - initial_state['q_w'] * data['q_z'] + initial_state['q_y'] * data['q_x'] - data['q_y'] * initial_state['q_x']

# 將相對四元數歸一化
magnitude = (relative_q_w**2 + relative_q_x**2 + relative_q_y**2 + relative_q_z**2)**0.5
data['relative_q_w'] = relative_q_w / magnitude
data['relative_q_x'] = relative_q_x / magnitude
data['relative_q_y'] = relative_q_y / magnitude
data['relative_q_z'] = relative_q_z / magnitude

# 將資料輸出至data_org.csv
if load_gt:
    output_path = "data_org.csv"
else:
    output_path = "dgvins_org.csv"
data.to_csv(output_path, index=False, header=None, float_format='%.9f')
