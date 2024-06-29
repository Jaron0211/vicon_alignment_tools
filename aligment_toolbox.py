import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

pd.options.display.float_format = '{:.4f}'.format

def MakeInterpolationData(source:pd.DataFrame, target:pd.DataFrame) -> pd.DataFrame :

    if (source['timestamp'].isna().any()) or (target['timestamp'].isna().any()):
        return source, target
    
    CacheDataframe = pd.DataFrame([],columns=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])

    CacheDataframe['timestamp'] = target['timestamp'].astype(int)
    TargetTimestamp = CacheDataframe['timestamp'].to_list()

    CacheDataframe = pd.concat([source, CacheDataframe]).drop_duplicates(subset='timestamp')
    CacheDataframe = CacheDataframe.sort_values(by=['timestamp'],ignore_index=True)

    CacheDataframe.interpolate(method='cubic',order=3, inplace=True, limit_direction='both')
    CacheDataframe.interpolate(method='linear',order=3, inplace=True, limit_direction='both')

    return CacheDataframe[CacheDataframe['timestamp'].isin(TargetTimestamp)]

if __name__ == '__main__':
    source = pd.read_csv('Aligned_dynamic_gt_1.csvedit.csv.csv', header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])
    target = pd.read_csv('Aligned_dynaVINS_50_dynamic_1.bag_vio.csv.csv', header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])

    df = MakeInterpolationData(source, target)
    print(df)

    plt.figure()
    plt.scatter(source['timestamp'].values, source['py'].values,s=1, c='r')
    plt.scatter(df['timestamp'].values, df['py'].values,s=2, c='g')
    plt.scatter(target['timestamp'].values, target['py'].values,s=1, c='b')
    [plt.axvline(x=x,linewidth=.1) for x in target['timestamp'].values]
    plt.show()
                
    
