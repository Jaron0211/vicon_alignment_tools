import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as ani

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

def RecoverRotation(vec1: np.ndarray , vec2: np.ndarray):
    from scipy.spatial.transform import Rotation as R
    axis = np.cross(vec1, vec2)

    # Compute the angle of rotation
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    # Normalize the axis
    axis = axis / np.linalg.norm(axis)

    # Create the rotation object using axis-angle representation
    rotation = R.from_rotvec(angle * axis)

    # Convert the rotation to a rotation matrix
    rotation_matrix = rotation.as_matrix()

    # Extract the Euler angles from the rotation matrix
    # Note: We assume a 'ZYX' rotation order (yaw, pitch, roll)

    euler_angles = rotation.as_euler('zyx', degrees=True)

    return rotation_matrix, euler_angles

def GetRotFromTwoPC(source: pd.DataFrame, target: pd.DataFrame) :
    # remove outliers
    from scipy import stats
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.distance import cdist
    import random

    TargetCache = target.copy()
    SourceCache = source.copy()

    TargetCache[['px','py','pz']] -= TargetCache[['px','py','pz']].mean()
    SourceCache[['px','py','pz']] -= SourceCache[['px','py','pz']].mean()

    #TargetCache[['px','py','pz']] -= TargetCache[['px','py','pz']].loc[1]
    #SourceCache[['px','py','pz']] -= SourceCache[['px','py','pz']].loc[1]

    # ax.scatter(TargetCache['px'], TargetCache['py'], s=.1, c='b')
    # ax.scatter(SourceCache['px'], SourceCache['py'], s=.1, c='r')

    def _closest_point(point, points):
        """ Find closest point from a list of points. """
        return points[cdist([point], points).argmin()]

    def _match_value(df, col1, x, col2):
        """ Match value x from col1 row to value in col2. """
        return df[df[col1] == x][col2].values[0]

    global MeanEular, previousEular, cosdiff, itercounter, momentum

    itercounter = 0
    cosdiff = 1
    MeanEular = np.array([0,0,0])
    momentum = np.array([0,0,0])
    previousEular = MeanEular
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')

    def __animation(i):

        global MeanEular, previousEular, cosdiff, itercounter, momentum

        SampleNum = 100
        
        ax.clear()
        ax.set_xlim3d(-2,2)
        ax.set_ylim3d(-2,2)
        ax.set_zlim3d(-2,2)
        rotated_coordinates_xyz = np.column_stack((TargetCache['px'], TargetCache['py'], TargetCache['pz']))

        rotation_xyz = R.from_euler('ZYX', MeanEular, degrees=True)
        rotated_coordinates_xyz = rotation_xyz.apply(rotated_coordinates_xyz)

        TargetCache[['px','py','pz']] = rotated_coordinates_xyz

        SourceCachePoint = pd.DataFrame()
        TargetCachePoint = pd.DataFrame()

        SourceCachePoint['point'] = [(x,y,z) for  x,y,z in zip(SourceCache['px'], SourceCache['py'], SourceCache['pz'])]
        TargetCachePoint['point'] = [(x,y,z) for  x,y,z in zip(TargetCache['px'][::int(TargetCache.shape[0]/SampleNum)], 
                                                               TargetCache['py'][::int(TargetCache.shape[0]/SampleNum)], 
                                                               TargetCache['pz'][::int(TargetCache.shape[0]/SampleNum)])]

        TargetCachePoint['closest'] = [_closest_point(x, list(SourceCachePoint['point'])) for x in TargetCachePoint['point']]

        ax.scatter3D(SourceCache['px'], SourceCache['py'], SourceCache['pz'], s=.1, c='r')
        ax.scatter3D(TargetCache['px'], TargetCache['py'], TargetCache['pz'], s=.1, c='b')

        Euler_list = []
        Distance_list = []

        for row in TargetCachePoint.iloc:
            ax.plot3D((row['point'][0], row['closest'][0]), (row['point'][1], row['closest'][1]), (row['point'][2], row['closest'][2]))
            RotM, Euler = RecoverRotation(np.array(row['point']), np.array(row['closest']))
            Distance = (row['point'][0] - row['closest'][0]), (row['point'][1] - row['closest'][1]), (row['point'][2] - row['closest'][2])
            Euler_list.append(Euler)
            Distance_list.append(Distance)
        
        weight = (Distance_list/np.sum(Distance_list))
        MeanEular = (np.mean((np.array(Euler_list)),axis=0))
        
        cosdiff = np.linalg.norm(MeanEular) / np.linalg.norm(previousEular)

        previousEular = MeanEular

        if cosdiff > 0.99:
            itercounter += 1
        else:
            itercounter = 0

    worker = ani.FuncAnimation(fig, __animation, interval = 10, frames=range(0,60))
    writergif = ani.PillowWriter(fps=15) 
    #plt.show()
    worker.save('ani.gif',writer=writergif)
    
   

        
def PlotDistribution(source: pd.DataFrame):

    from statistics import geometric_mean

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(source['px'].values, source['py'].values,source['pz'].values, s=.5, c='r')

    SourceX = source['px'].to_numpy() + source['px'].min()+1
    SourceY = source['py'].to_numpy() + source['py'].min()+1
    SourceZ = source['pz'].to_numpy() + source['pz'].min()+1

    XMean = geometric_mean(list(SourceX.tolist())) - (source['px'].min()+1)
    YMean = geometric_mean(list(SourceY.tolist())) - (source['py'].min()+1)
    ZMean = geometric_mean(list(SourceZ.tolist())) - (source['pz'].min()+1)

    XStd = SourceX.std()
    YStd = SourceY.std()
    ZStd = SourceZ.std()

    Coefs = (XStd*2,YStd*2,ZStd*2)
    Rx, Ry, Rz = 1/np.sqrt(Coefs) # Radii corresponding to the coefficients

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    ex = Rx * np.outer(np.cos(u), np.sin(v)) + XMean
    ey = Ry * np.outer(np.sin(u), np.sin(v)) + YMean
    ez = Rz * np.outer(np.ones_like(u), np.cos(v)) + ZMean

    #ax.plot_surface(ex, ey, ez,  rstride=4, cstride=4, color='b')
    #ax.scatter(XMean, YMean, ZMean, s=1, c='b')

    #plt.show()

    return XMean, YMean, ZMean

def CreateTimeViaHz(source: pd.DataFrame, start_time, hz = 100):

    SourceCache = source.copy()

    SourceCache['timestamp'] = [ (start_time + 1/hz*i*10**9) for i in range(0, source.shape[0], 1)]

    return SourceCache

def AlignmentPath(source: pd.DataFrame, target: pd.DataFrame):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    SourceCache = source.copy()

    if (source['timestamp'].isna().any()):
        SourceCache = CreateTimeViaHz(source, target['timestamp'].loc[0])
        print('Make Interpolation Timestamp')

    ErrorDistribution = []


    def _animate(i):

        
        SourceShift = SourceCache.copy()
        SourceShift['timestamp'] += i*10**9/10
        if SourceShift['timestamp'].min() > target['timestamp'].max(): return
        SourceCacheInterpolation = MakeInterpolationData(SourceShift,target)

        ax.clear()

        ax.scatter(SourceCacheInterpolation['timestamp'],SourceCacheInterpolation['px'], s=.5, c='b')
        ax.axvline(SourceShift['timestamp'][0],c='g')
        ax.scatter(target['timestamp'],target['px'], s=.5, c='r')

        Error = abs(SourceCacheInterpolation['px'].to_numpy() - target['px'].to_numpy()).sum()
        print(Error)
        ErrorDistribution.append(Error)

        return [ax]

    worker = ani.FuncAnimation(fig, _animate, frames = range(SourceCache['timestamp'].shape[0]), interval=1, repeat = False)
    plt.show()

    fig2 = plt.figure()
    plt.plot(range(len(ErrorDistribution)), ErrorDistribution)
    plt.scatter(ErrorDistribution.index(min(ErrorDistribution)), min(ErrorDistribution))
    plt.text(ErrorDistribution.index(min(ErrorDistribution)), min(ErrorDistribution), str(ErrorDistribution.index(min(ErrorDistribution))))
    plt.show()

    fig3 = plt.figure()
    plt.scatter(SourceCache['timestamp']+ErrorDistribution.index(min(ErrorDistribution))*10**9/10 ,SourceCache['px'], s=.5, c='b')
    plt.scatter(target['timestamp'],target['px'], s=.5, c='r')
    plt.show()
    


if __name__ == '__main__':
    # source = pd.read_csv('Aligned_dynamic_gt_1.csvedit.csv.csv', header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])
    source = pd.read_csv('csv/gt/dynamic_gt_4.csvedit.csv', header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])
    target = pd.read_csv('csv/dgvins/dgvins_dynamic_4.bag_vio.csv', header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])


    #AlignmentPath(source, target)

    GetRotFromTwoPC(source, target)
    
