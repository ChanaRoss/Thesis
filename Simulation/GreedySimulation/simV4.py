import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import copy, time, sys
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import imageio
from scipy.stats import truncnorm
import pickle

import more_itertools as mrt

sns.set()


# functions
def insertTimeEvent(timeLine, t):
    """
    inserts an event into the time line. maintains sorted time line.
    if t is allready in timeline, nothing is added.
    :param timeLine: list of integer time events
    :param t: event time
    :return: new timeLine
    """
    if t in timeLine:
        return timeLine
    else:
        timeLine.append(t)
        timeLine.sort()
        return timeLine


def manhattenPath(position1, position2):
    """
    calculates grid deltas between two positions relative to
    first position
    :param position1:
    :param position2:
    :return: dx (int), dy (int)
    """
    dx = position2[0] - position1[0]
    dy = position2[1] - position1[1]
    return dx, dy


def manhattenDist(position1, position2):
    """
    calc manhatten distacne between two positions
    :param position1:
    :param position2:
    :return: distance (int)
    """
    dx, dy = manhattenPath(position1, position2)
    return float(abs(dx) + abs(dy))


def moveCar(car, dt):
    """
    moves car towards target, random dx or dy
    :param car: car dict that matches template
    :return: car with updates position
    """
    dist2Target = manhattenDist(car['position'], car['target'])
    dx, dy = manhattenPath(car['position'], car['target'])
    possibleDistance = dt * car['velocity']
    # possible distance should not be greater than distance to target.
    assert (
                possibleDistance <= dist2Target or dist2Target == 0.)  # this would indicate an event missing from the timeLine
    if possibleDistance == dist2Target:
        car['position'] = car['target']
        return car
    elif possibleDistance < dist2Target:
        if np.random.binomial(1, 0.5):
            ix = np.min([possibleDistance, abs(dx)]) * np.sign(dx)
            iy = np.max([possibleDistance - abs(ix), 0]) * np.sign(dy)
        else:
            iy = np.min([possibleDistance, abs(dy)]) * np.sign(dy)
            ix = np.max([possibleDistance - abs(iy), 0]) * np.sign(dx)
        # update position
        car['position'][0] += ix
        car['position'][1] += iy
    return car


def timeLineUpdate(carDict, timeLine, currentTimeIndex):
    """
    checks if event must be added, adds if necessary
    assumes current time index is not the last
    :param carDict:
    :param timeLine:
    :return: updates timeline
    """
    currentDelta = timeLine[currentTimeIndex + 1] - timeLine[currentTimeIndex]
    timeDeltas = [manhattenDist(c['position'], c['target']) / c['velocity'] for c in carDict.values() if
                  c['target'] is not None]
    timeDeltas = [t for t in timeDeltas if t != 0.0 if t != 0]
    minDelta = np.min(timeDeltas) if len(timeDeltas) > 0 else currentDelta + 1
    # add event
    if minDelta < currentDelta:
        timeLine = insertTimeEvent(timeLine, minDelta + timeLine[currentTimeIndex])
    return timeLine


def createCostMatrix(carDict, eventDict, currentTime):
    """
    create a cost matrix of matching car i with event j. uses manhatten
    distance as cost
    :param carDict: all cars in simulation dict
    :param eventDict: event dictionary
    :param currentTime: current point in timeLine
    :return: index to id dict, cost matrix
    """
    # crete index to id dict for events
    index2id = {i: e['id'] for i, e in enumerate(eventDict.values())}
    id2index = {v: k for k, v in index2id.items()}

    # create matrix
    numEvents = len(eventDict)
    numCars = len(carDict)
    mat = np.zeros(shape=(numCars, numEvents), dtype=np.int32)
    for c in carDict.values():
        for e in eventDict.values():
            tempDist = manhattenDist(c['position'], e['position'])
            mat[c['id'], id2index[e['id']]] = tempDist
    return index2id, mat


def unifromRandomEvents(simLength, n):
    return list(np.random.randint(0, simLength, n).astype(dtype=np.float32))


def plotSim(carDict, eventDict, currentTime, maxX, maxY):
    """
    plot cars as red points, events as blue points,
    and lines connecting cars to their targets
    :param carDict:
    :param eventDict:
    :return: None
    """
    # filter events
    filteredEvents = filterEvents(eventDict, currentTime)
    # scatter
    carPositions = [c['position'] for c in carDict.values()]
    evePositions = [e['position'] for e in filteredEvents.values()]
    plt.scatter([p[0] for p in carPositions], [p[1] for p in carPositions], c='r', label='cars', alpha=0.5)
    for car in carDict.values():
        plt.text(car['position'][0], car['position'][1] + 0.1, str(car['id']))
    plt.scatter([p[0] for p in evePositions], [p[1] for p in evePositions], c='b', label='events', alpha=0.5)

    for event in filteredEvents.values():
        plt.text(event['position'][0], event['position'][1] + 0.1,
                 's.t: ' + str(np.round(event['timeStart'], 1)) + ',ID: ' + str(event['id']))

    # lines
    for c in carDict.values():
        if c['target'] is None:
            continue
        else:
            plt.plot([c['position'][0], c['target'][0]], [c['position'][1], c['target'][1]], 'g--')
    plt.title('current time: {0}'.format(currentTime))
    plt.grid(True)
    plt.legend()
    plt.ylim([-3, maxY + 3])
    plt.xlim([-3, maxX + 3])


def plotSimForGif(carDict, eventDict, currentTime, maxX, maxY):
    """
        plot cars as red points, events as blue points,
        and lines connecting cars to their targets
        :param carDict:
        :param eventDict:
        :return: image for gif
        """
    # filter events
    filteredEvents = filterEvents(eventDict, currentTime)
    # scatter
    carPositions = [c['position'] for c in carDict.values()]
    evePositions = [e['position'] for e in filteredEvents.values()]
    fig, ax = plt.subplots()
    ax.scatter([p[0] for p in carPositions], [p[1] for p in carPositions], c='r', label='cars', alpha=0.5)
    for car in carDict.values():
        ax.text(car['position'][0], car['position'][1] + 0.1, str(car['id']))
    ax.scatter([p[0] for p in evePositions], [p[1] for p in evePositions], c='b', label='events', alpha=0.5)

    for event in filteredEvents.values():
        ax.text(event['position'][0], event['position'][1] + 0.1,
                's.t: ' + str(np.round(event['timeStart'], 1)) + ',ID: ' + str(event['id']))

    # lines
    for c in carDict.values():
        if c['target'] is None:
            continue
        else:
            ax.plot([c['position'][0], c['target'][0]], [c['position'][1], c['target'][1]], 'g--')
    ax.set_title('current time: {0}'.format(currentTime))
    ax.grid(True)
    ax.legend()
    ax.set_ylim([-3, maxY + 3])
    ax.set_xlim([-3, maxX + 3])

    # Used to return the plot as an image rray
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def createCarPositionLog(car, currentTime):
    return {'position': copy.deepcopy(car['position']), 'time': currentTime, 'target': copy.deepcopy(car['target']), 'targetId': copy.deepcopy(car['targetId'])}


def createCarUseLog(carDict, currentTime):
    inUse = len([c for c in carDict.values() if c['target'] is not None])
    notInUse = len(carDict) - inUse
    return {'occupied': inUse, 'idle': notInUse}


def createEventLog(eventLog, eventDict, currentTime):
    eventLog['count'].append(len([e for e in eventDict.values() if e['timeStart'] <= currentTime]))
    eventLog['closed'].append(len([e for e in eventDict.values() if e['closed']]))
    eventLog['canceled'].append(len([e for e in eventDict.values() if not e['closed'] and e['timeEnd'] < currentTime]))
    eventLog['current'].append(len(filterEvents(eventDict, currentTime)))
    eventLog['time'].append(currentTime)
    for event in eventDict.values():
        if event['timeStart'] <= currentTime and event['closed'] == False:
            eventDict[event['id']]['statusLog'].append([currentTime, True])
        else:
            eventDict[event['id']]['statusLog'].append([currentTime, False])
    return eventLog


def filterEvents(eventDict, currentTime):
    return {e['id']: e for e in eventDict.values() if not e['closed'] if e['timeStart'] <= currentTime if
            e['timeEnd'] > currentTime}


def main():
    flagLoadParam = 1
    imageList = []
    # templates
    carEntityTemp = {'id': 0, 'velocity': 0, 'position': [0, 0], 'target': None, 'targetId': None, 'finished': 0}
    eventTemp = {'waitTime': 0 ,'position': [], 'timeStart': 0, 'timeEnd': 0, 'closed': False, 'statusLog': [], 'id': 0, 'prob': 1}
    # params
    numEvents = 20
    lengthSim = 20  # minutes
    numCars = 2
    gridWidth = 5
    gridHeight = 5

    eventLoc = []
    eventTimes = []
    eventLog = {'count': [], 'closed': [], 'canceled': [], 'current': [], 'time': []}
    if flagLoadParam == 1:
        pickleName = 'log_Cost_WaitTime_CarMovement_5grid_2cars_20simLengh_30StochasticLength_3Prediction_1aStarWeight'
        lg=pickle.load(open('/home/chanaby/Documents/Thesis/Thesis/Simulation/Anticipitory/Results/' + pickleName + '.p', 'rb'))
        # init event dict
        eventDict = {}
        # take event times from file -
        eventDataTuple = [(e['timeStart'], e['position']) for e in lg['events'].values()]
        # create event dict
        for i, et in enumerate(eventDataTuple):
            event = copy.deepcopy(eventTemp)
            event['timeStart'] = eventDataTuple[i][0]
            event['timeEnd'] = lengthSim
            # event['timeEnd'] = et + int(np.random.randint(1, np.min([maxEventDuration, lengthSim-et]), 1))
            event['position'] = eventDataTuple[i][1]
            event['id'] = i
            eventDict[i] = event
        # logs
        carPositionLog = {}
        for i in range(numCars):
            carPositionLog[i] = []
        carUseLog = {}
        # car dictionary
        carDict = {}
        carPos = [c['position'] for c in lg['carDict'].values()]
        for i in range(numCars):
            car = copy.deepcopy(carEntityTemp)
            car['position'] = carPos[i]
            car['velocity'] = 1
            car['id'] = i
            carDict[i] = car
    else:
        # set seed
        np.random.seed(1)
        maxEventDuration = lengthSim
        lastEventDelta = 1
        # initialize structures
        # logs
        carPositionLog = {}
        for i in range(numCars):
            carPositionLog[i] = []
        carUseLog = {}
        # car dictionary
        carDict = {}
        for i in range(numCars):
            car = copy.deepcopy(carEntityTemp)
            car['position'] = [int(np.random.randint(0, gridWidth, 1)), int(np.random.randint(0, gridHeight, 1))]
            car['velocity'] = 1
            car['id'] = i
            carDict[i] = car

        # init event dict
        eventDict = {}
        # randomize event times
        eventTimes = unifromRandomEvents(lengthSim - lastEventDelta, numEvents)
        locX = gridWidth / 2
        scaleX = gridWidth
        locY = gridHeight / 2
        scaleY = gridHeight
        eventPosX = truncnorm.rvs((0 - locX) / scaleX, (gridWidth - locX) / scaleX, loc=locX, scale=scaleX,
                                  size=numEvents).astype(np.int64)
        eventPosY = truncnorm.rvs((0 - locY) / scaleY, (gridHeight - locY) / scaleY, loc=locY, scale=scaleY,
                                  size=numEvents).astype(np.int64)
        # create event dict
        for i, et in enumerate(eventTimes):
            event = copy.deepcopy(eventTemp)
            event['timeStart'] = et
            event['timeEnd'] = et + np.min([maxEventDuration, lengthSim - et])
            # event['timeEnd'] = et + int(np.random.randint(1, np.min([maxEventDuration, lengthSim-et]), 1))
            event['position'] = [eventPosX[i], eventPosY[i]]
            event['id'] = i
            eventDict[i] = event

    # init timeLine
    timeLine = list({et['timeStart'] for et in eventDict.values()}) + [lengthSim]
    timeLine.sort()

    # initial state plot
    # plotSim(carDict, eventDict, 0, gridWidth, gridHeight)
    imageList.append(plotSimForGif(carDict, eventDict, 0, gridWidth, gridHeight))

    # main loop
    timeIndex = 0
    while timeIndex < len(timeLine) - 1:
        # log car positions and close events
        for car in carDict.values():
            carPositionLog[car['id']].append(createCarPositionLog(car, timeLine[timeIndex]))
            # close target event if it is reached
            if (isinstance(car['target'], list)) and (car['position'] == car['target']):
                eventDict[car['targetId']]['closed'] = True
                car['finished'] += 1
            # reset car target
            car['target'] = None
            car['targetId'] = None

        # log events
        eventLog = createEventLog(eventLog, eventDict, timeLine[timeIndex])

        # filter events
        filteredEvents = filterEvents(eventDict, timeLine[timeIndex])

        # if no events exist, incriment and continue
        if len(filteredEvents) == 0:
            # plot simulation
            # plotSim(carDict, eventDict, timeLine[timeIndex], gridWidth, gridHeight)
            imageList.append(plotSimForGif(carDict, eventDict, timeLine[timeIndex], gridWidth, gridHeight))
            timeIndex += 1
        else:
            for event in filteredEvents.values():
                eventDict[event['id']]['waitTime'] += 1
            # calculate cost matrix
            index2id, costMat = createCostMatrix(carDict, filteredEvents, timeLine[timeIndex])

            # matching
            carIndices, matchedIndices = linear_sum_assignment(costMat)

            # update targets
            for ci, ri in zip(carIndices, matchedIndices):
                carDict[ci]['target'] = copy.deepcopy(eventDict[index2id[ri]]['position'])
                carDict[ci]['targetId'] = index2id[ri]

            # update timeLine
            timeLineUpdate(carDict, timeLine, timeIndex)

            # car use log
            carUseLog[timeLine[timeIndex]] = createCarUseLog(carDict, timeLine[timeIndex])

            # # plot simulation
            # # plotSim(carDict, eventDict, timeLine[timeIndex], gridWidth, gridHeight)
            imageList.append(plotSimForGif(carDict, eventDict, timeLine[timeIndex], gridWidth, gridHeight))

            # move cars
            for cid in carDict:
                if carDict[cid]['target'] is not None:
                    carDict[cid] = moveCar(carDict[cid], timeLine[timeIndex + 1] - timeLine[timeIndex])

            # incriment timeIndex
            timeIndex += 1
        # dump logs
    with open('HungarianMethod_log_Cost_WaitTime_CarMovement_' + str(gridHeight) + 'grid_' + str(numCars) + 'cars_' + str( lengthSim) + 'simLengh.p', 'wb') as out:
        pickle.dump({'cars': carPositionLog, 'events': eventDict, 'eventLog': eventLog, 'carDict': carDict}, out)
    kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
    imageio.mimsave('./gif_HungarianMethod_log_Cost_WaitTime_CarMovement_' + str(gridHeight) + 'grid_' + str(
        numCars) + 'cars_' + str(lengthSim) + 'simLengh.gif', imageList, fps=1)
    # simulation summary
    plt.close('all')
    plotingTimeline = timeLine[:-1]
    plt.figure(2)
    plt.plot(plotingTimeline, eventLog['count'], c='r', label='Number of created events')
    plt.plot(plotingTimeline, eventLog['closed'], c='b', label='Number of closed events')
    # plt.plot(plotingTimeline, eventLog['canceled'], c='y', label='canceled')
    plt.legend()
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('num events')
    plt.title('events over time')

    # current over time
    plt.figure(3)
    plt.plot(plotingTimeline, eventLog['current'], label='current')
    plt.title('currently open over time')
    plt.xlabel('time')
    plt.ylabel('num events')
    plt.grid(True)

    # car barplot
    plt.figure(4)
    eventsPerCar = {c['id']: c['finished'] for c in carDict.values()}
    plt.bar(eventsPerCar.keys(), eventsPerCar.values())
    plt.title('events handeled per car')
    plt.xlabel('car id')
    plt.ylabel('num events')
    plt.grid(True)

    plt.figure(5)
    for event in eventDict.values():
        if event['closed']:
            plt.scatter(event['position'][0], event['position'][1], color='g', label=event['id'])
            plt.text(event['position'][0], event['position'][1], event['id'])
        else:
            plt.scatter(event['position'][0], event['position'][1], color='r', label=event['id'])
            plt.text(event['position'][0], event['position'][1], event['id'])
    plt.xlabel('grid X')
    plt.ylabel('grid Y')
    plt.title('event locations')
    plt.xlim([-3, gridWidth + 3])
    plt.ylim([-3, gridHeight + 3])
    plt.show()
    return


if __name__ == '__main__':
    main()
    print('Done.')
