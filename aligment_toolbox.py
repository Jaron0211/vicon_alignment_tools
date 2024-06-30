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

def CalculateCenterPoint(target:pd.DataFrame) -> list:

    MeanPoints = [target['px'].mean(), target['py'].mean(), target['pz'].mean()]
    return MeanPoints

def GetRotFromTwoPC(source: pd.DataFrame, target: pd.DataFrame) :
    # remove outliers
    from scipy import stats

    plt.figure()
    TargetCache = target.copy()

    plt.scatter(TargetCache['px'], TargetCache['py'], s=.1, c='b')

    TargetCacheOutliers = TargetCache[ abs(TargetCache['px'].diff() - TargetCache['px'].diff().mean()) > TargetCache['px'].diff().std()]

    plt.scatter(TargetCacheOutliers['px'], TargetCacheOutliers['py'], s=.2, c='r')
    
    #plt.scatter(TargetCache['timestamp'], TargetCache['px'].diff())

    plt.show()

        
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
    #source = pd.read_csv('Aligned_dynamic_gt_1.csvedit.csv.csv', header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])
    source = pd.read_csv('csv/gt/dynamic_gt_4.csvedit.csv', header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])
    target = pd.read_csv('csv/vins-fusion/vins_dynamic_4.bag_vio.csv', header=None, names=['timestamp','px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz' ])

    # df = MakeInterpolationData(source, target)
    # print(df)

    # plt.figure()
    # plt.scatter(source['timestamp'].values, source['py'].values,s=1, c='r')
    # plt.scatter(df['timestamp'].values, df['py'].values,s=2, c='g')
    # plt.scatter(target['timestamp'].values, target['py'].values,s=1, c='b')
    # [plt.axvline(x=x,linewidth=.1) for x in target['timestamp'].values]
    

    # plt.figure()

    SourceMeanPoint = CalculateCenterPoint(source)
    TargetMeanPoint = CalculateCenterPoint(target)
    # plt.scatter(source['px'].values, source['py'].values, s=1, c='r')
    # plt.scatter(SourceMeanPoint[0],SourceMeanPoint[1], s=5, c='g')

    # plt.scatter(target['px'].values, target['py'].values, s=1, c='b')
    # plt.scatter(TargetMeanPoint[0],TargetMeanPoint[1], s=5, c='y')

    TError = [ TargetMeanPoint[0] - SourceMeanPoint[0], TargetMeanPoint[1] - SourceMeanPoint[1], TargetMeanPoint[2] - SourceMeanPoint[2]]
    TargetMoved = target.copy()
    TargetMoved['px'] -= TError[0]
    TargetMoved['py'] -= TError[1]
    TargetMoved['pz'] -= TError[2]
    # plt.scatter(TargetMoved['px'].values, TargetMoved['py'].values, s=1, c='c')

    # plt.show()

    #GetRotFromTwoPC(source, TargetMoved)

    #XMean, YMean, ZMean = PlotDistribution(source)

    #AlignmentPath(source, target)

    GetRotFromTwoPC(source, target)
    
