import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as ani

from scipy import stats
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import random

pd.options.display.float_format = '{:.4f}'.format

def MakeInterpolationData(source:pd.DataFrame, target:pd.DataFrame) -> pd.DataFrame :

    if (source['timestamp'].isna().any()) or (target['timestamp'].isna().any()):
        return source, target
    
    CacheDataframe = pd.DataFrame([],columns=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])

    CacheDataframe['timestamp'] = target['timestamp'].astype(int)
    TargetTimestamp = CacheDataframe['timestamp'].to_list()

    CacheDataframe = pd.concat([source, CacheDataframe]).drop_duplicates(subset='timestamp')
    CacheDataframe = CacheDataframe.sort_values(by=['timestamp'],ignore_index=True)

    #CacheDataframe.interpolate(method='cubic',order=3, inplace=True, limit_direction='both')
    CacheDataframe.interpolate(method='linear',order=3, inplace=True, limit_direction='both')

    return CacheDataframe

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

        SampleNum = 30
        
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
        #TargetCachePoint['farest'] = [_farest_point(x, list(SourceCachePoint['point'])) for x in TargetCachePoint['point']]

        ax.scatter3D(SourceCache['px'], SourceCache['py'], SourceCache['pz'], s=.1, c='r')
        ax.scatter3D(TargetCache['px'], TargetCache['py'], TargetCache['pz'], s=.1, c='b')

        Euler_list = []
        Distance_list = []

        for row in TargetCachePoint.iloc:
            ax.plot3D((row['point'][0], row['closest'][0]), (row['point'][1], row['closest'][1]), (row['point'][2], row['closest'][2]), c='r')
            #ax.plot3D((row['point'][0], row['farest'][0]), (row['point'][1], row['farest'][1]), (row['point'][2], row['farest'][2]), c='b')
            
            RotM, Euler = RecoverRotation(np.array(row['point']), np.array(row['closest']))
            Distance = (row['point'][0] - row['closest'][0]), (row['point'][1] - row['closest'][1]), (row['point'][2] - row['closest'][2])
            Euler_list.append(Euler)
            # RotM, Euler = RecoverRotation(np.array(row['point']), np.array(row['farest']))
            # Euler_list.append(Euler)
            Distance_list.append(Distance)
        
        weight = (Distance_list/np.sum(Distance_list))
        MeanEular = (np.mean((np.array(Euler_list)),axis=0))
        
        cosdiff = np.linalg.norm(MeanEular-previousEular)

        previousEular = MeanEular

        if cosdiff > 0.99:
            itercounter += 1
        else:
            itercounter = 0

    worker = ani.FuncAnimation(fig, __animation, interval = 10, frames=range(0,60))
    writergif = ani.PillowWriter(fps=15) 
    #plt.show()
    worker.save('RT_force.gif',writer=writergif)

def GeometryDescriptor(source: pd.DataFrame):

    from scipy.spatial.transform import Rotation as R

    SourceCache = source.copy()

    MeanPoint = SourceCache[['px','py','pz']].mean().to_numpy()

    RotM, Eular = RecoverRotation(np.array(MeanPoint - MedianPoint), np.array(MeanPoint - MeanPoint))

    SourceCache[['px','py','pz']] = rotated_coordinates_xyz

    Eular_list = []
    for row in SourceCache.iloc:

        RotM, Eular = RecoverRotation(np.array(row[['px','py','pz']]), np.array(MeanPoint))
        Eular = tuple(Eular.tolist())

        Eular = tuple(row[['px','py','pz']].to_list())
        Eular_list.append(Eular)

    RotOfPath = np.array(Eular_list).mean()
    rotation_xyz = R.from_euler('zyx', RotOfPath)

    rotated_coordinates_xyz = np.column_stack((SourceCache['px'], SourceCache['py'], SourceCache['pz']))
    rotated_coordinates_xyz = rotation_xyz.apply(rotated_coordinates_xyz)
    
    SourceCache['descriptor'] = Eular_list
    SourceCache[['px','py','pz']] = source[['px','py','pz']]

    return SourceCache

def _closest_point(point, points):
        """ Find closest point from a list of points. """
        return points[cdist([point], points).argmin()]

def ICP(source: pd.DataFrame, 
        target: pd.DataFrame, 
        SampleNum : int = 60, 
        Rot_pre : np.array = np.array([[0,0,0],[0,0,0],[0,0,0]])) :
    
    Rot_pre = Rot_pre

    TargetCache = target.copy()
    SourceCache = source.copy()

    SourceCachePoint = pd.DataFrame()
    TargetCachePoint = pd.DataFrame()

    TargetCacheCentroid = np.asarray(TargetCache[['px','py','pz']].mean())
    SourceCacheCentroid = np.asarray(SourceCache[['px','py','pz']].mean())

    SourceCachePoint['point'] = [[x,y,z] for  x,y,z in zip(SourceCache['px'], SourceCache['py'], SourceCache['pz'])]
    TargetCachePoint['point'] = [[x,y,z] for  x,y,z in zip(TargetCache['px'][::int(TargetCache.shape[0]/SampleNum)], 
                                                            TargetCache['py'][::int(TargetCache.shape[0]/SampleNum)], 
                                                            TargetCache['pz'][::int(TargetCache.shape[0]/SampleNum)])]
    
    TargetCachePoint['closest'] = [_closest_point(x, list(SourceCachePoint['point'])) for x in TargetCachePoint['point']]

    SourcePointMatrix = np.asarray([list(point) for point in TargetCachePoint['closest'].to_list()])
    TargetPointMatrix = np.asarray([list(point) for point in TargetCachePoint['point'].to_list()])

    CovMatrix = TargetPointMatrix.transpose() @ SourcePointMatrix

    #SVD
    U, X, Vt = np.linalg.svd(CovMatrix)
    Rot = U @ Vt
    
    Trans = TargetCacheCentroid - Rot @ SourceCacheCentroid
    Trans = np.reshape(Trans, (1,3))

    # rotated_coordinates_xyz = np.column_stack((TargetCache['px'], TargetCache['py'], TargetCache['pz']))

    # rotation_xyz = R.from_matrix(Rot.T)
    # rotated_coordinates_xyz = rotation_xyz.apply(rotated_coordinates_xyz)

    # TargetCache[['px','py','pz']] = rotated_coordinates_xyz - Trans
    # TargetCache[['px','py','pz']] += SourceCache[['px','py','pz']].loc[0] - TargetCache[['px','py','pz']].loc[0]

    return Rot, Trans

        
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

def CreateTimeViaHz(source: pd.DataFrame, end_time, hz = 100):

    SourceCache = source.copy()

    timestamp_array = [ (end_time - 1/hz*i*10**9) for i in range(0, source.shape[0], 1)]

    SourceCache['timestamp'] = timestamp_array[::-1]

    return SourceCache

def AlignmentPath(source: pd.DataFrame, target: pd.DataFrame):
    '''
    Source: The GT file
    Target: The Path file
    '''

    SourceCache = source.copy()
    TargetCache = target.copy()

    if (source['timestamp'].isna().any()):
        SourceCache = CreateTimeViaHz(source, target['timestamp'].loc[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Before Iteration')
    ax.scatter(SourceCache['timestamp'], SourceCache['px'], s= .1, c = 'b')
    ax.scatter(TargetCache['timestamp'], TargetCache['px'], s= .1, c = 'r')
    plt.show()

    BestTimeShift = 0
    Pre_BestTimeShift = 10**100
    PreviousError = 10**9

    print('First Proximation')

    _StartOfSecondIteration = 0
    _EndOfSecondIteration = int(TargetCache['timestamp'].max() - TargetCache['timestamp'].min() + SourceCache['timestamp'].max() - SourceCache['timestamp'].min())
    _Step = int((_EndOfSecondIteration - _StartOfSecondIteration)/20)

    while 1:

        for i in tqdm(range(_StartOfSecondIteration, _EndOfSecondIteration, _Step)):

            SourceShift = SourceCache.copy()

            SourceShift['timestamp'] += i

            SourceCacheInterpolation = MakeInterpolationData(SourceShift,TargetCache)

            Error = abs(SourceCacheInterpolation['px'][SourceCacheInterpolation['timestamp'].isin(target['timestamp'])].to_numpy() - target['px'].to_numpy()).sum()
            if Error < PreviousError:

                BestTimeShift = i
                PreviousError = Error
            
    
        _StartOfSecondIteration = int(BestTimeShift - _Step)
        _EndOfSecondIteration = int(BestTimeShift + _Step)
        _Step = int((_EndOfSecondIteration - _StartOfSecondIteration)/20)

        print(1 - abs( BestTimeShift / Pre_BestTimeShift))
        if 1 - abs( BestTimeShift / Pre_BestTimeShift) == 0:
            print('Best time shift: ', BestTimeShift)
            break

        Pre_BestTimeShift = BestTimeShift
    
    SourceCache['timestamp'] += BestTimeShift
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('After Iteration')
    ax.scatter(SourceCache['timestamp'], SourceCache['px'], s= .1, c = 'b')
    ax.scatter(TargetCache['timestamp'], TargetCache['px'], s= .1, c = 'r')
    plt.show()

    return SourceCache, TargetCache
    

if __name__ == '__main__':
    source = pd.read_csv('Aligned_dynamic_gt_1.csvedit.csv.csv', header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])
    target = pd.read_csv('Aligned_dynaVINS_50_dynamic_1.bag_vio.csv.csv', header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])
    
    source['timestamp'] = np.nan
    #source = pd.read_csv('csv/gt/dynamic_gt_3.csvedit.csv', header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])
    #target = pd.read_csv('csv/dgvins/dgvins_dynamic_3.bag_vio.csv', header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])

    source, target = AlignmentPath(source, target)

    #GetRotFromTwoPC(source, target)
    # ICP(source, target)
    #GeometryDescriptor(source)
    
