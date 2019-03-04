import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
from pyproj import Proj,transform
from collections import Counter
import pickle



def longLatToXY(long, lat):
    """
    converts epsg:4326 long lat (degrees) coordiantes to epsg:3857 xy coordiantes
    :param long: longtitude
    :param lat: latitude
    :return: xy tuple
    """
    sourceProj = Proj(init='epsg:4326')
    targetProj = Proj(init='epsg:3857')
    return transform(sourceProj, targetProj, long, lat)


def XyToLongLat(x, y):
    """
    converts epsg:3856 xy coordinates to epsg:4326 long lat coordiantes (degrees)
    :param x: x coordinate
    :param y: y coordiante
    :return: long lat tuple
    """
    sourceProj = Proj(init='epsg:3857')
    targetProj = Proj(init='epsg:4326')
    return transform(sourceProj, targetProj, x, y)


def main():
    # set path to dataset (csv)
    dataPath = '/Users/chanaross/Documents/Thesis/UberData/allData.csv'
    xGridResolution = 250  # grid rectangle width
    yGridResolution = 250  # grid rectangle height
    # read data to dataframe
    df = pd.read_csv(dataPath)

    # date manipulation
    df['TimeStamp'] = pd.DatetimeIndex(df['Date/Time'])
    df['weekday'] = df['TimeStamp'].dt.weekday
    df['weeknum'] = df['TimeStamp'].dt.week
    df['month'] = df['TimeStamp'].dt.month
    df['hour'] = df['TimeStamp'].dt.hour
    df['minute'] = df['TimeStamp'].dt.minute
    # remove string datetime, keep stamp
    df = df.drop('Date/Time', axis=1)
    # add repetitive weekly time id
    df['weekPeriod'] = df['weekday']*(24*4) + df['hour']*4 + np.floor_divide(df['minute'], 15).astype(np.int64)
    df['dayQuad'] = np.floor_divide(df['hour'], 4).astype(np.int64)

    # coordinate transformation and grid creation
    # transform picture to be straight with lat , lon (transformation angle is 36.1)

    df['Fixed_Lon'] = df['Lon'] * np.cos(36.1 * np.pi / 180) - df['Lat'] * np.sin(36.1 * np.pi / 180)
    df['Fixed_Lat'] = df['Lat'] * np.cos(36.1 * np.pi / 180) + df['Lon'] * np.sin(36.1 * np.pi / 180)

    df['Lon'] = df['Fixed_Lon']
    df['Lat'] = df['Fixed_Lat']
    # create list of tuples of long lat coordiantes
    longLatTupleList = zip(df['Lon'].tolist(), df['Lat'].tolist())
    # create list of tuples of x,y coordiantes
    xyTupleList = [longLatToXY(*t) for t in longLatTupleList]
    # add to df
    xCoordinates = [c[0] for c in xyTupleList]
    yCoordinates = [c[1] for c in xyTupleList]
    df['x'] = xCoordinates
    df['y'] = yCoordinates
    # create grid
    xMinCoord = np.min(df['x'])
    yMinCoord = np.min(df['y'])
    # calculate distance from grid edges
    df['x_grid_dist'] = df['x']-xMinCoord
    df['y_grid_dist'] = df['y']-yMinCoord
    # add grid indices to dataframe
    df['grid_x'] = np.floor_divide(df['x_grid_dist'], xGridResolution).astype(np.int64)
    df['grid_y'] = np.floor_divide(df['y_grid_dist'], yGridResolution).astype(np.int64)
    # add single index grid id
    maxXgrid = np.max(df['grid_x'])
    df['grid_id'] = df['grid_x'] + df['grid_y']*maxXgrid

    # pickle data
    df.to_pickle(dataPath.replace('.csv', 'LatLonCorrected_Gridpickle250.p'))
    with open (dataPath.replace('.csv', 'LatLonCorrected__Gridpickle250.p'), 'wb') as op:
        pickle.dump(df,op)

    df.to_csv(dataPath.replace('.csv', 'LatLonCorrected_GridXY2.csv'))
    return

if __name__=='__main__':
    main()
    print('Done.')