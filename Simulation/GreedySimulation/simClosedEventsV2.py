import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import copy, time, sys
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import imageio
from scipy.stats import truncnorm
import pickle

# my files
sys.path.insert(0, '/home/chana/Documents/Thesis/FromGitFiles/Simulation/Anticipatory/')
from offlineOptimizationProblem_TimeWindow import runMaxFlowOpt

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


def moveCar(car, dt, totalCost):
    """
    moves car towards target, random dx or dy
    :param car: car dict that matches template
    :return: car with updates position
    """
    dist2Target = manhattenDist(car['position'], car['target'])
    dx, dy = manhattenPath(car['position'], car['target'])
    possibleDistance = dt * car['velocity']
    # possible distance should not be greater than distance to target.
    assert (possibleDistance <= dist2Target or dist2Target == 0.)  # this would indicate an event missing from the timeLine
    if possibleDistance == dist2Target:
        car['position'] = car['target']
        totalCost += abs(possibleDistance)
        return car,totalCost
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
        totalCost += abs(ix)+abs(iy)
    return car,totalCost


def timeLineUpdate(carDict, timeLine, currentTimeIndex):
    """
    checks if event must be added, adds if necessary
    assumes current time index is not the last
    :param carDict:
    :param timeLine:
    :return: updates timeline
    """
    currentDelta = timeLine[currentTimeIndex + 1] - timeLine[currentTimeIndex]
    timeDeltas = [manhattenDist(c['position'], c['target']) / c['velocity'] for c in carDict.values() if c['target'] is not None]
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
    plt.legend()
    plt.ylim([-3, maxY + 3])
    plt.xlim([-3, maxX + 3])
    # Major ticks every 1,
    major_ticks = np.arange(-3, maxX + 3, 1)
    plt.xticks(major_ticks)
    plt.yticks(major_ticks)
    plt.grid(True)


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
    ax.legend()
    ax.set_ylim([-3, maxY + 3])
    ax.set_xlim([-3, maxX + 3])
    # Major ticks every 1
    major_ticks = np.arange(-3, maxX + 3, 1)

    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)

    # And a corresponding grid
    ax.grid(True)
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


def createEventLog(eventLog, eventDict, currentTime,totalCost,eventPenalty):
    eventLog['count'].append(len([e for e in eventDict.values() if e['timeStart'] <= currentTime]))
    eventLog['answered'].append(len([e for e in eventDict.values() if e['answered']]))
    eventLog['canceled'].append(len([e for e in eventDict.values() if e['canceled']]))
    eventLog['current'].append(len(filterEvents(eventDict, currentTime)))
    eventLog['time'].append(currentTime)
    for event in eventDict.values():
        if event['timeStart'] <= currentTime and event['timeEnd'] > currentTime and not event['answered']:
            eventDict[event['id']]['statusLog'].append([currentTime, True])
        else:
            eventDict[event['id']]['statusLog'].append([currentTime, False])
        if event['timeEnd'] == currentTime and not event['answered']:
            eventDict[event['id']]['canceled'] = True
            totalCost += eventPenalty
    return eventLog,totalCost


def filterEvents(eventDict, currentTime):
    return {e['id']: e for e in eventDict.values() if not e['answered'] if e['timeStart'] <= currentTime if e['timeEnd'] > currentTime}


def main():
    flagLoadParam = 1
    imageList = []
    # templates
    carEntityTemp = {'id': 0, 'velocity': 0, 'position': [0, 0], 'target': None, 'targetId': None, 'finished': 0}
    eventTemp = {'waitTime': 0 ,'position': [], 'timeStart': 0, 'timeEnd': 0, 'answered': False,'canceled':False, 'statusLog': [], 'id': 0, 'prob': 1}
    totalCost      = 0
    lastEventDelta = 1
    eventLog = {'count': [], 'answered': [], 'canceled': [], 'current': [], 'time': []}
    if flagLoadParam == 1:
        pickleName = 'SimAnticipatoryMioFinalResults_15numEvents_2numCars_0.75lam_7gridSize'
        pathName = '/home/chana/Documents/Thesis/FromGitFiles/Simulation/Anticipitory/PickleFiles/'
        lg = pickle.load(open(pathName + pickleName + '.p', 'rb'))
        # params
        lengthSim       = np.max(lg['time'])  # minutes
        gridWidth       = lg['gridSize']
        gridHeight      = lg['gridSize']
        eventReward     = lg['pathresults'][0].closeReward
        eventPenalty    = lg['pathresults'][0].cancelPenalty
        carsPos         = [c.path for c in lg['pathresults'][0].cars.notCommited.values()]
        numCars         = len(carsPos)
        eventsPos       = [c.position for c in lg['pathresults'][0].events.notCommited.values()]
        numEvents       = len(eventsPos)
        eventsStartTime = [c.startTime for c in lg['pathresults'][0].events.notCommited.values()]
        eventsEndTime   = [c.endTime for c in lg['pathresults'][0].events.notCommited.values()]
        # init event dict
        eventDict = {}
        # create event dict
        for i, in range(numEvents):
            event = copy.deepcopy(eventTemp)
            event['timeStart'] = eventsStartTime[i]
            event['timeEnd']   = eventsEndTime[i]
            event['position']  = list(eventsPos[i])
            event['id']        = i
            eventDict[i]       = event
        # logs
        carPositionLog = {}
        for i in range(numCars):
            carPositionLog[i] = []
        carUseLog = {}
        # car dictionary
        carDict   = {}
        for i in range(numCars):
            car             = copy.deepcopy(carEntityTemp)
            car['position'] = list(carsPos[i])
            car['velocity'] = 1
            car['id']       = i
            carDict[i]      = car
    else:
        # params
        numEvents = 40
        lengthSim = 35  # minutes
        numCars = 2
        gridWidth = 9
        gridHeight = 9
        eventReward = 10
        timeWindow = 3
        eventPenalty = 100
        eventLoc = []
        eventTimes = []
        pickleName = 'Test_nCars'+str(numCars)+'_nEvents'+str(numEvents)
        # set seed
        np.random.seed(1)
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
        scaleX = gridWidth/3
        locY = gridHeight / 2
        scaleY = gridHeight/3
        eventPosX = truncnorm.rvs((0 - locX) / scaleX, (gridWidth - locX) / scaleX, loc=locX, scale=scaleX,
                                  size=numEvents).astype(np.int64)
        eventPosY = truncnorm.rvs((0 - locY) / scaleY, (gridHeight - locY) / scaleY, loc=locY, scale=scaleY,
                                  size=numEvents).astype(np.int64)
        # create event dict
        for i, et in enumerate(eventTimes):
            event = copy.deepcopy(eventTemp)
            event['timeStart'] = et
            event['timeEnd'] = et + timeWindow
            # event['timeEnd'] = et + int(np.random.randint(1, np.min([maxEventDuration, lengthSim-et]), 1))
            event['position'] = [eventPosX[i], eventPosY[i]]
            event['id'] = i
            eventDict[i] = event

    # init timeLine
    timeLine = list({et['timeStart'] for et in eventDict.values()}) + [lengthSim]
    timeLine.sort()
    lengthSim = np.max(timeLine)
    deltaT = 1
    continuesTimeLine = np.linspace(0, lengthSim+lastEventDelta, (lengthSim + lastEventDelta)/ deltaT + 1)

    # initial state plot
    # plotSim(carDict, eventDict, 0, gridWidth, gridHeight)
    imageList.append(plotSimForGif(carDict, eventDict, 0, gridWidth, gridHeight))

    # main loop
    timeIndex = 0

    while timeIndex < len(continuesTimeLine) - 1:
        # log car positions and close events
        for car in carDict.values():
            carPositionLog[car['id']].append(createCarPositionLog(car, continuesTimeLine[timeIndex]))
            # close target event if it is reached
            if (isinstance(car['target'], list)) and (car['position'] == car['target']):
                eventDict[car['targetId']]['answered'] = True
                eventDict[car['targetId']]['waitTime'] =  continuesTimeLine[timeIndex] - eventDict[car['targetId']]['timeStart']
                car['finished'] += 1
                totalCost -= eventReward
            # reset car target
            car['target'] = None
            car['targetId'] = None

        # log events
        eventLog,totalCost = createEventLog(eventLog, eventDict, continuesTimeLine[timeIndex],totalCost,eventPenalty)

        # filter events
        filteredEvents = filterEvents(eventDict, continuesTimeLine[timeIndex])

        # if no events exist, incriment and continue
        if len(filteredEvents) == 0:
            # plot simulation
            # plotSim(carDict, eventDict, timeLine[timeIndex], gridWidth, gridHeight)
            imageList.append(plotSimForGif(carDict, eventDict, continuesTimeLine[timeIndex], gridWidth, gridHeight))
            timeIndex += 1
        else:
            # calculate cost matrix
            index2id, costMat = createCostMatrix(carDict, filteredEvents, continuesTimeLine[timeIndex])

            # matching
            carIndices, matchedIndices = linear_sum_assignment(costMat)

            # update targets
            for ci, ri in zip(carIndices, matchedIndices):
                carDict[ci]['target'] = copy.deepcopy(eventDict[index2id[ri]]['position'])
                carDict[ci]['targetId'] = index2id[ri]

            # update timeLine
            # timeLineUpdate(carDict, timeLine, timeIndex)
            # deltaT = timeLine[timeIndex + 1] - timeLine[timeIndex]
            for event in filteredEvents.values():
                eventDict[event['id']]['waitTime'] += deltaT
                totalCost += deltaT
            # car use log
            carUseLog[continuesTimeLine[timeIndex]] = createCarUseLog(carDict, continuesTimeLine[timeIndex])

            # # plot simulation
            # # plotSim(carDict, eventDict, timeLine[timeIndex], gridWidth, gridHeight)
            imageList.append(plotSimForGif(carDict, eventDict, continuesTimeLine[timeIndex], gridWidth, gridHeight))

            # move cars
            for cid in carDict:
                if carDict[cid]['target'] is not None:
                    carDict[cid],totalCost = moveCar(carDict[cid], continuesTimeLine[timeIndex + 1] - continuesTimeLine[timeIndex],totalCost)

            # incriment timeIndex
            timeIndex += 1
        # dump logs
    with open('HungarianMethod'+pickleName+'.p', 'wb') as out:
        pickle.dump({'cars'     : carPositionLog,
                     'events'   : eventDict,
                     'eventLog' : eventLog,
                     'carDict'  : carDict,
                     'cost'     : totalCost}, out)
    print('total cost is:' + str(totalCost))
    kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
    imageio.mimsave('./gif_HungarianMethod'+pickleName+'.gif', imageList, fps=1)
    # simulation summary
    plt.close('all')
    plotingTimeline = continuesTimeLine[:-1]
    plt.figure(2)
    plt.plot(plotingTimeline, eventLog['count'], c='r', label='Number of created events')
    plt.plot(plotingTimeline, eventLog['answered'], c='b', label='Number of answered events')
    plt.plot(plotingTimeline, eventLog['canceled'], c='y', label='canceled')
    plt.legend()
    MajorTicks = np.arange(0,np.max(plotingTimeline)+2,2)
    plt.xticks(MajorTicks)
    plt.yticks(MajorTicks)
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
    MajorTicks = np.arange(0, np.max(plotingTimeline) + 2, 2)
    plt.xticks(MajorTicks)
    plt.yticks(MajorTicks)
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
        if event['answered']:
            plt.scatter(event['position'][0], event['position'][1], color='g', label=event['id'])
            plt.text(event['position'][0], event['position'][1], event['id'])
        else:
            plt.scatter(event['position'][0], event['position'][1], color='r', label=event['id'])
            plt.text(event['position'][0], event['position'][1], event['id'])
    plt.xlabel('grid X')
    plt.ylabel('grid Y')
    plt.title('event locations')
    plt.xlim([-3, gridWidth  + 3])
    plt.ylim([-3, gridHeight + 3])
    MajorTicks = np.arange(-3,gridHeight+3,1)
    plt.xticks(MajorTicks)
    plt.yticks(MajorTicks)
    plt.show()
    return


if __name__ == '__main__':
    main()
    print('Done.')
