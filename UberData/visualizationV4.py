import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import cm
import matplotlib.animation as animation
matplotlib.use('TKAgg')
import pandas as pd
from collections import Counter
import seaborn as sns


def gridIndicesToMatIndices(i,j, maxI, maxJ):
    """
    converts grid coordinates to numpy array coordiantes
    grid has (0,0) at bottom left corner, numpy has (0,0)
    at top left corner.
    :param i: x - column index
    :param j: y - row index
    :param maxI: number of columns in grid
    :param maxJ: number of rows in grid
    :return: converted row and column coordiantes
    """
    return (i-1,maxJ-j-1)

def createBarPlot(df, block=True):
    # create barplot graph
    for wnum in np.unique(df['weeknum']):
        tdf = df[df['weeknum'] == wnum]
        tcnt = Counter(tdf['weekPeriod'])
        plt.bar(tcnt.keys(), tcnt.values(), label=wnum, alpha=0.5)
    plt.legend()
    plt.show(block=block)
    return

def createHeatMaps(df, block=True):
    mapDimX = np.max(df['grid_x'])
    mapDimY = np.max(df['grid_y'])
    for q in range(3):
        for wnum in np.unique(df['weeknum']):
            fig,ax =plt.subplots(nrows=2, ncols=4)
            for wday in range(7):
                tempMat = np.zeros(shape=(mapDimY, mapDimX))
                tempDf = df[(df['weekday'] == wday) & (df['weeknum'] == wnum) & (df['dayQuad'] == q)]
                for index, row in tempDf.iterrows():
                    c, r = gridIndicesToMatIndices(row['grid_x'], row['grid_y'], mapDimX, mapDimY)
                    tempMat[r, c] += 1
                # tempMat.dump('Mat_'+'weekNum_'+str(wnum)+'q_'+str(q)+'wday_'+str(wday)+'.p')
                # add to subplot
                ax[wday//4][wday%4].set_title("quad: {0}, week: {1}, day: {2}".format(q, wnum, wday))
                sns.heatmap(tempMat, ax=ax[wday//4][wday%4])
            plt.show(block=block)

def CreateMatrix(df):
    mapDimX = np.max(df['grid_x'])
    mapDimY = np.max(df['grid_y'])
    weekNum = 18
    weekDay = 2

    for hour in range(24):
        tempMat = np.zeros(shape=(mapDimY, mapDimX,24))
        tempDf = df[(df['weekday'] == weekDay) & (df['weeknum'] == weekNum)]
        for index, row in tempDf.iterrows():
            c, r = gridIndicesToMatIndices(row['grid_x'], row['grid_y'], mapDimX, mapDimY)
            tempMat[r, c,row['hour']] += 1

    tempMat.dump('3DMat_' + 'weekNum_' + str(weekNum) + 'wday_' + str(weekDay) + '.p')



def createStackPlot(df, block=True):
    # stackplot
    x = np.linspace(0, (4 * 24 * 7) - 1, (4 * 24 * 7) - 1)
    for wnum in np.unique(df['weeknum']):
        print(wnum)
        seriesStackList = []
        labels = []
        # create color map matrix
        mapDimX = np.max(df['grid_x'])
        mapDimY = np.max(df['grid_y'])
        cMat = np.zeros(shape=(mapDimY, mapDimX, 3))
        # create color map for plots
        cMap = sns.color_palette("coolwarm", np.unique(df['grid_id']).size)
        for c, gid in zip(cMap, np.unique(df['grid_id'])):
            # create data vector
            tempVector = np.zeros_like(x, dtype=np.int64)
            # count orders per weekPeriod
            tempDf = df[(df['grid_id'] == gid) & (df['weeknum']==wnum)]
            wkprd, cnt = np.unique(tempDf['weekPeriod'], return_counts=True)
            indVec = wkprd.astype(np.int64) - 1
            # add counts to vector
            tempVector[indVec] = cnt
            # add vector to list
            seriesStackList.append(tempVector)
            # add label
            labels.append(gid)

        # sort labels and vectors according to max
        vectorMaxValues = [np.max(v) for v in seriesStackList]
        vectorSortIndices = np.argsort(vectorMaxValues)
        sortedLabels = [labels[i] for i in vectorSortIndices]
        sortedVectors = [seriesStackList[i] for i in vectorSortIndices]

        # add color to color matrix
        for i,gid in enumerate(sortedLabels):
            mc, mr = gridIndicesToMatIndices(gid % mapDimX, gid // mapDimX, mapDimX, mapDimY)
            cMat[mr, mc, :] = np.array(cMap[i])

        # plot
        fig = plt.figure()
        fig.suptitle('weeknum: {0}, numSeries: {1}'.format(wnum, len(seriesStackList)))
        ax1 = fig.add_subplot(121)
        ax1 = plt.stackplot(x, np.vstack(sortedVectors), colors=cMap, baseline='zero', labels=labels)
        ax2 = fig.add_subplot(122)
        ax2 = plt.imshow(cMat)
    plt.show(block=block)

def createDayHistograms(df, block=True):
    # preprare dict of lists for sum of orders aggregated by weekday
    dayData = {}
    for d in range(7):
        dayData[d] = []
    # fill day data dict
    for wnum in np.unique(df['weeknum']):
        print('weeknum: {0}'.format(wnum))
        tempDf = df[df['weeknum']==wnum]
        tempCounter = Counter(tempDf['weekday'])
        for k in dayData.keys():
            dayData[k].append(tempCounter[k])
    # create figure
    fig,ax = plt.subplots(nrows=2, ncols=4)
    fig.suptitle('distributions of orders per day')
    for d in range(7):
        ax[d//4][d%4].set_title("day: {0}, orders per day".format(d))
        ax[d//4][d%4].hist(dayData[d], bins=20)
        ax[d // 4][d % 4].set_xlim(left=0, right=45000)
    plt.show()

def createScatterMaps(df, block=True):
    fig = plt.figure()
    plt.style.use('dark_background')
    scs = []
    for wnum in np.unique(df['weeknum']):
        for day in range(7):
            tempDf = df[(df['weeknum'] == wnum) & (df['weekday'] == day)]
            sc = plt.scatter(tempDf['Lon'], tempDf['Lat'], c='white', alpha=0.6, s=0.02, animated=True)
            scs.append([sc])
            #plt.xlim([-74.1, -73.9])
            #plt.ylim([40.6, 40.9])
            plt.xlim([np.min(df['Lon']), np.max(df['Lon'])])
            plt.ylim([np.min(df['Lat']), np.max(df['Lat'])])
    ani = animation.ArtistAnimation(fig, scs, interval=400, blit=True, repeat_delay=1000)
    plt.show()
    return

def createStackedBarPlot(df, block=True):
    weekPeriodCounts = np.zeros(shape=(np.max(df['weekPeriod']+1,)))
    vectorX = np.linspace(0, np.max(df['weekPeriod']), np.max(df['weekPeriod']+1)).astype(np.int64)
    plt.figure()
    for wnum in np.unique(df['weeknum']):
        # create filtered df
        tempDf = df[df['weeknum']==wnum]
        # get counts for weekperiod
        vals,counts = np.unique(df['weekPeriod'], return_counts=True)
        # create y vector for plot
        tempY = np.zeros_like(vectorX).astype(np.int64)
        tempY[vals] = counts
        plt.bar(vectorX, tempY, bottom=weekPeriodCounts, label=wnum)
        # update bottom vector for stacked chart
        weekPeriodCounts[vals] += counts
    plt.xlabel("weekPeriod")
    plt.ylabel("order count")
    plt.legend()
    # plt.show(block=block)
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.show(block=block)
    # plt.savefig('StackedBarPlot' + '.png')
    return

def createFastStackPlot(df ,block=True):
    # dimensions for maps
    mapDimX = np.max(df['grid_x'])
    mapDimY = np.max(df['grid_y'])
    for wnum in np.unique(df['weeknum']):
        # filter dataset
        tempDf = df[df['weeknum']==wnum]
        # pivot
        tempTb = pd.pivot_table(tempDf, index='grid_id', columns='weekPeriod', values='count', aggfunc=np.sum).fillna(0)
        tempTbColumns = tempTb.columns.tolist()
        # add max column
        tempTb['max'] = tempTb.max(axis=1)
        # sort the table
        tempTb = tempTb.sort_values(by='max')
        # convert to matrix
        gridIdMat = tempTb.drop(labels='max', axis=1).values
        # create color pallet
        maxValues = sorted(list(np.unique(tempTb['max'])))
        cPal = sns.color_palette('Reds', np.unique(tempTb['max']).size)
        colorDict = {maxValues[i] : cPal[i] for i in range(len(maxValues))}
        cmap = [colorDict[v] for v in tempTb['max']]

        # create heat matrix
        heatMat = np.zeros(shape=(mapDimY, mapDimX, 3))
        for i,gid in enumerate(tempTb.index.tolist()):
            mc, mr = gridIndicesToMatIndices(gid % mapDimX, gid // mapDimX, mapDimX, mapDimY)
            heatMat[mr,mc,:] = np.array(cmap[i])
        # plot
        fig = plt.figure()
        fig.suptitle('weeknum: {0}, numSeries: {1}'.format(wnum, gridIdMat.shape[0]))
        ax1 = fig.add_subplot(121)
        ax1 = plt.stackplot(tempTbColumns, gridIdMat, colors=cmap, labels=tempTb.columns, baseline='zero')
        ax2 = fig.add_subplot(122)
        ax2 = plt.imshow(heatMat)
        plt.show()
    #     manager = plt.get_current_fig_manager()
    #     manager.window.showMaximized()

        plt.show(block=False)
        # plt.savefig('FastStackPlot_' + str(wnum) + '.png')

def createProbMatrix(df):
    dfTemp = df.copy()
    dfTemp = dfTemp[dfTemp['weekday'] == 1]  # use only monday
    dfTemp['grid_x'] = dfTemp['grid_x'] - np.min(dfTemp['grid_x'])
    dfTemp['grid_y'] = dfTemp['grid_y'] - np.min(dfTemp['grid_y'])
    gridX = dfTemp['grid_x'].unique()
    gridY = dfTemp['grid_y'].unique()

    mat = np.zeros(shape = (gridX.size,gridY.size,24,df['weeknum'].unique().size))
    wnumMin = df['weeknum'].min()
    weekDay = 1
    for wnum in df['weeknum'].unique():
        dfTemp1 = dfTemp[dfTemp['weeknum'] == wnum]
        for t in dfTemp1['hour'].unique():
            dfTemp2 = dfTemp1[dfTemp1['hour'] == t]
            for ix, iy in zip(dfTemp2['grid_x'], dfTemp2['grid_y']):
                mat[ix, iy, t, wnum-wnumMin] += 1
    maxNumEvents = np.max(mat).astype(int)
    matOut = np.zeros(shape = (gridX.size, gridY.size, 24, maxNumEvents+1))
    for ix in range(mat.shape[0]):
        for iy in range(mat.shape[1]):
            for t in range(mat.shape[2]):
                for nWeek in range(mat.shape[3]):
                    nEvents = mat[ix, iy, t, nWeek].astype(int)
                    #if nEvents > 20:
                    #    nEvents = 20
                    matOut[ix, iy, t, nEvents] += 1
                # normalizing numbers to be probability instead of absolute value
                matOut[ix, iy, t, :] = matOut[ix, iy, t, :]/np.sum(matOut[ix, iy, t, :])
    matOut.dump('4DProbabilityMat_' + 'wday_' + str(weekDay) + '.p')
    fig, ax = plt.subplots(1, 1)
    for i in range(matOut.shape[2]):
        a = np.sum(matOut[:, :, i, :], axis=(0, 1))
        ax.plot(range(a.shape[0]), a, label=str(i))
    plt.legend()
    plt.show()
    print('hi')
    return

def main():
    # path to data pickle (after preproc)
    dataPath = '/Users/chanaross/dev/Thesis/UberData/alldata_corrected_pickle.p'
    # dataPath = '/Users/chanaross/Documents/Thesis/uberAnalysis/allData.p'
    # read data
    df = pd.read_pickle(dataPath)
    df['count'] = 1

    # filter data
    # df = df[(df["Lon"]>(-74.1)) & (df['Lon']<(-73.9))]
    # df = df[(df["Lat"]>40.6) & (df['Lat']<40.876)]
    maxDimX = df['grid_x'].max()
    maxDimY = df['grid_y'].max()
    # filter file to only manhattan area (get rid of sparse area)
    df = df[(df["Lon"]>=(-83.812)) & (df['Lon']<=(-83.7668))]
    df = df[(df["Lat"]>=-16.483) & (df['Lat']<=-16.3314)]
    # add single index grid id
    maxXgrid = np.max(df['grid_x'])
    df['grid_id'] = df['grid_x'] + df['grid_y'] * maxXgrid

    createProbMatrix(df)
    # createFastStackPlot(df,True)
    # createScatterMaps(df,True)
    # createHeatMaps(df,False)
    # CreateMatrix(df)
    #createStackPlot(df, True)

    return

if __name__=='__main__':
    main()
    print('Done.')