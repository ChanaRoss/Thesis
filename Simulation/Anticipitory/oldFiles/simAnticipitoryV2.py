import numpy as np
from matplotlib import pyplot as plt
import copy,time,sys,pickle
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import imageio
from scipy.stats import truncnorm
import itertools as rt
sns.set()

def SetInputAttributes(object, params, **kwargs):
    for key in params.keys():
        setattr(object, 'm_'+key, params[key])
    for key in kwargs:
        if key in params.keys():
            setattr(object, 'm_'+key, kwargs[key])

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
    return dx,dy

def manhattenDist(position1, position2):
    """
    calc manhatten distacne between two positions
    :param position1:
    :param position2:
    :return: distance (int)
    """
    dx,dy = manhattenPath(position1, position2)
    return float(abs(dx) + abs(dy))

def moveCar(car, dt):
    """
    moves car towards target, random dx or dy
    :param car: car dict that matches template
    :return: car with updates position
    """
    dist2Target = manhattenDist(car['position'], car['target'])
    dx, dy = manhattenPath(car['position'], car['target'])
    possibleDistance = dt*car['velocity']
    totalDistance = 0
    # possible distance should not be greater than distance to target.
    if (possibleDistance<=dist2Target or abs(dist2Target)<0.0001) == False:
        print('stop')
    assert(possibleDistance<=dist2Target or abs(dist2Target)<0.0001) # this would indicate an event missing from the timeLine
    if possibleDistance==dist2Target:
        car['position'] = car['target']
        totalDistance = dist2Target
        return car,totalDistance
    elif possibleDistance<dist2Target:
        if np.random.binomial(1, 0.5):
            ix = np.min([possibleDistance, abs(dx)])*np.sign(dx)
            iy = np.max([possibleDistance-abs(ix), 0])*np.sign(dy)
        else:
            iy = np.min([possibleDistance, abs(dy)]) * np.sign(dy)
            ix = np.max([possibleDistance-abs(iy), 0]) * np.sign(dx)
        # update position
        car['position'][0] += ix; car['position'][1] += iy
        totalDistance = np.abs(ix) + np.abs(iy)
    return car,totalDistance

def timeLineUpdate(carDict, timeLine, currentTimeIndex):
    """
    checks if event must be added, adds if necessary
    assumes current time index is not the last
    :param carDict:
    :param timeLine:
    :return: updates timeline
    """
    currentDelta = timeLine[currentTimeIndex+1] - timeLine[currentTimeIndex]
    timeDeltas = [manhattenDist(c['position'], c['target'])/c['velocity'] for c in carDict.values() if c['target'] is not None]
    timeDeltas = [t for t in timeDeltas if t!=0.0 if t!=0]
    minDelta = np.min(timeDeltas) if len(timeDeltas)>0 else currentDelta+1
    # add event
    if minDelta<currentDelta:
        timeLine = insertTimeEvent(timeLine, minDelta+timeLine[currentTimeIndex])
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
    index2id = {i:e['id'] for i,e in enumerate(eventDict.values())}
    id2index = {v:k for k,v in index2id.items()}

    # create matrix
    numEvents = len(eventDict)
    numCars = len(carDict)
    mat = np.zeros(shape=(numCars, numEvents), dtype=np.int32)
    for c in carDict.values():
        for e in eventDict.values():
            tempDist = manhattenDist(c['position'], e['position'])
            mat[c['id'], id2index[e['id']]] = tempDist
    return index2id,mat

def unifromRandomEvents(startTime,endSimTime, n):
    return list(np.random.randint(startTime, endSimTime, n).astype(dtype=np.float32))

def createCarPositionLog(car, currentTime):
    return {'position': car['position'], 'time': currentTime, 'target': car['target'], 'targetId': car['targetId']}

def createCarUseLog(carDict, currentTime):
    inUse = len([c for c in carDict.values() if c['target'] is not None])
    notInUse = len(carDict)-inUse
    return {'occupied': inUse, 'idle': notInUse}

def createEventLog(eventLog, eventDict, currentTime):
    eventLog['count'].append(len([e for e in eventDict.values() if e['timeStart']<=currentTime]))
    eventLog['closed'].append(len([e for e in eventDict.values() if e['closed']]))
    eventLog['canceled'].append(len([e for e in eventDict.values() if not e['closed'] and e['timeEnd']<currentTime]))
    eventLog['current'].append(len(filterEvents(eventDict, currentTime)))
    eventLog['time'].append(currentTime)
    for event in eventDict.values():
        if event['timeStart'] <= currentTime and event['closed'] == False:
            eventDict[event['id']]['statusLog'].append([currentTime, True])
        else:
            eventDict[event['id']]['statusLog'].append([currentTime, False])
    return eventLog

def filterEvents(eventDict, currentTime):
    return {e['id']: e for e in eventDict.values() if not e['closed'] if e['timeStart'] <= currentTime if e['timeEnd']>currentTime}

def CalculateCostForKnownEventList(carDict,eventDict,timeLine,gridWidth,gridHeight):
    # main loop
    timeIndex = 0

    numCars = len(carDict)

    # initialize structures
    # logs
    carPositionLog = {}
    for i in range(numCars):
        carPositionLog[i] = []
    eventLogInner = {'count': [], 'closed': [], 'canceled': [], 'current': [],'time':[]}
    carUseLog = {}
    TotalCost = 0
    while timeIndex < len(timeLine) - 1:
        # log car positions and close events
        for car in carDict.values():
            carPositionLog[car['id']].append(createCarPositionLog(car, timeLine[timeIndex]))
            if timeIndex == 0:
                for event in eventDict.values():
                    if car['position'] == event['position']:
                        eventDict[event['id']]['closed'] = True
                        eventDict[event['id']]['waitTime'] = timeLine[timeIndex] - event['timeStart']
                        car['finished'] +=1
            # close target event if it is reached
            else:
                if (isinstance(car['target'], list)) and (car['position'] == car['target']):
                    eventDict[car['targetId']]['closed'] = True
                    eventDict[event['id']]['waitTime'] = timeLine[timeIndex] - event['timeStart']
                    car['finished'] += 1
            # reset car target
            car['target'] = None
            car['targetId'] = None

        # log events
        eventLogInner = createEventLog(eventLogInner, eventDict, timeLine[timeIndex])

        # filter events
        filteredEvents = filterEvents(eventDict, timeLine[timeIndex])

        # if no events exist, incriment and continue
        if len(filteredEvents) == 0:
            timeIndex += 1
        else:
            # calculate cost matrix
            index2id, costMat = createCostMatrix(carDict, filteredEvents, timeLine[timeIndex])
            # matching
            carIndices, matchedIndices = linear_sum_assignment(costMat)

            # update targets
            for ci, ri in zip(carIndices, matchedIndices):
                carDict[ci]['target'] = eventDict[index2id[ri]]['position']
                carDict[ci]['targetId'] = index2id[ri]

            # update timeLine
            timeLineUpdate(carDict, timeLine, timeIndex)

            # car use log
            carUseLog[timeLine[timeIndex]] = createCarUseLog(carDict, timeLine[timeIndex])

            # move cars
            for cid in carDict:
                if carDict[cid]['target'] is not None:
                    carDict[cid],carMovementDistance = moveCar(carDict[cid], timeLine[timeIndex + 1] - timeLine[timeIndex])
                    TotalCost = TotalCost + carMovementDistance  # sums up the total sum of the cost of moving the cars
            # incriment timeIndex
            timeIndex += 1
    # for each event add up the time he waited (accumilative add) and add the last bit from whole minute he waited
    totalWaitTimeOfEvents = np.sum([np.sum(range(int(e['waitTime'])))+(e['waitTime']-int(e['waitTime'])) for e in eventDict.values()])
    percentageEventsClosed = 100.0 * eventLogInner['closed'][-1] / eventLogInner['count'][-1]

    return TotalCost,percentageEventsClosed,totalWaitTimeOfEvents

def buildEventList(startTime,endSimTime,lastEventDelta,numEventsForList,gridWidth,gridHeight,eventTemp):
    # init event dict
    eventDict = {}
    locX = gridWidth / 2
    scaleX = gridWidth
    locY = gridHeight / 2
    scaleY = gridHeight
    # randomize event times
    eventTimes = unifromRandomEvents(startTime,endSimTime - lastEventDelta, numEventsForList)
    eventPosX = truncnorm.rvs((0 - locX) / scaleX, (gridWidth - locX) / scaleX, loc=locX, scale=scaleX,
                              size=numEventsForList).astype(np.int64)
    eventPosY = truncnorm.rvs((0 - locY) / scaleY, (gridHeight - locY) / scaleY, loc=locY, scale=scaleY,
                              size=numEventsForList).astype(np.int64)

    # create event dict
    for i, et in enumerate(eventTimes):
        event = copy.deepcopy(eventTemp)
        event['timeStart'] = et
        event['timeEnd'] = endSimTime
        event['waitTime'] = event['timeEnd'] - event['timeStart']
        event['position'] = [eventPosX[i], eventPosY[i]]
        event['id'] = i
        eventDict[i] = event
    # init timeLine (last time is the length of the simulation wanted)
    timeLine = list({et['timeStart'] for et in eventDict.values()}) + [endSimTime]
    timeLine.sort()

    return timeLine,eventDict



def main():
    # define starting positions in the simulation:
    # set seed
    seed = 1
    np.random.seed(seed)
    # params
    epsilon = 0.1 # distance between locations to be considered same location
    numCars = 2
    numEvents = 70
    lengthSim = 50  # minutes
    gridWidth = 10
    gridHeight = 10
    eventDuration = lengthSim
    lastEventDelta = 3
    numStochasticEvents = 280
    # templates
    carEntityTemp = {'id': 0, 'velocity': 0, 'position': [0, 0], 'target': None, 'targetId': None, 'finished': 0}
    eventTemp = {'position': [], 'timeStart': 0, 'timeEnd': 0, 'closed': False, 'id': 0, 'prob': 1, 'statusLog': [], 'waitTime': 0}

    # initialize action dictionary - assuming at each deltaTime one step is taken (can change based on velocity of each car
    actionOptions = {'Up': [0, 1], 'Down': [0, -1], 'Left': [-1, 0], 'Right': [1, 0]}

    # initialize car dictionary
    carDict = {}
    actionChosen = []  # this is the action chosen from all possible actions for each car
    # initialize stochastic dictionaries and variables
    stochasticEventDict = {}
    stochasticTimeLine = {}
    # initialize logs
    carPositionLog = {}
    for i in range(numCars):
        carPositionLog[i] = []
    eventLog = {'count': [], 'closed': [], 'canceled': [], 'current': [],'time':[]}
    carUseLog = {}
    # define car dictionary that will actually move in the simulation
    for i in range(numCars):
        car = copy.deepcopy(carEntityTemp)
        car['position'] = [int(np.random.randint(0, gridWidth, 1)), int(np.random.randint(0, gridHeight, 1))]
        car['velocity'] = 1
        car['id'] = i
        carDict[i] = car
    # init event dict
    startTime = 0
    kwargs ={'startTime'              : startTime,
             'endSimTime'             : lengthSim,
             'lastEventDelta'         : lastEventDelta,
             'numEventsForList'       : numEvents,
             'gridWidth'              : gridWidth,
             'gridHeight'             : gridHeight,
             'eventTemp'              : eventTemp}
    timeLine, eventDict = buildEventList(**kwargs)

    # init simulation parameters -
    timeIndex = 0
    # initilize stochastic event list for each set checked -
    lengthPrediction = lengthSim
    # number of events for prediction is the number of events in simulation divided by the simulation time, for same amount per hour
    numEventsForPrediction = lengthPrediction*(numEvents//lengthSim)
    deltaT = 1
    continuesTimeLine = np.linspace(0, lengthSim+lastEventDelta, (lengthSim + lastEventDelta)/ deltaT + 1)
    costPerActionList = []
    while timeIndex < len(continuesTimeLine) - 1:
        iterStart = time.clock()
        costOfAction = 999999999
        # find all events that are not closed and already started :
        filteredEvents = filterEvents(eventDict, continuesTimeLine[timeIndex])
        # log car positions and close events
        for car in carDict.values():
            tempFilteredEvents = copy.deepcopy(filteredEvents)
            if len(tempFilteredEvents) !=0:
                for event in tempFilteredEvents.values():
                    if np.linalg.norm(np.array(event['position']) - np.array(car['position']), 2) < epsilon:  # this car picked up this event
                        eventDict[event['id']]['closed'] = True
                        eventDict[event['id']]['waitTime'] = continuesTimeLine[timeIndex] - event['timeStart']
                        car['targetId'] = event['id']
                        car['finished'] += 1
                        filteredEvents.pop(event['id'])
            carPositionLog[car['id']].append(createCarPositionLog(car, continuesTimeLine[timeIndex]))
            # reset car target
            car['target'] = None
            car['targetId'] = None
        # log events
        eventLog = createEventLog(eventLog, eventDict, continuesTimeLine[timeIndex])

        # if no events exist, incriment and continue
        if len(filteredEvents) == 0:
            timeIndex += 1
        else:
            numOptions = 4**(len(carDict))
            costPerAction = np.zeros(shape=(numOptions,numStochasticEvents+1))
            for j,actions in enumerate(rt.product(actionOptions.values(), repeat=len(carDict))):
                carDictForCalc = {}
                FlagOutOfGrid = False
                # create dictionary of cars for the stochastic calculations to find the cost of each action combination
                for i,car in enumerate(carDict.values()):
                    carWithActioin = car.copy()
                    carWithActioin['position'] = [a + b for a,b in zip(carWithActioin['position'],actions[i])]
                    carDictForCalc[car['id']] = carWithActioin
                    if np.abs(carWithActioin['position'][0])>gridWidth or np.abs(carWithActioin['position'][1])>gridHeight:
                        FlagOutOfGrid = True
                        carIndexOutOfGrid = car['id']
                if FlagOutOfGrid:
                    print('action out of bounds for car:'+str(carIndexOutOfGrid))
                    continue
                # calculate the cost of n eventLists for each action combination to find the expected value of moving in these actions
                stochasticCost = []
                for i in range(numStochasticEvents):
                    # events should start after the events that are happening now
                    kwargs = {'startTime'               : continuesTimeLine[timeIndex] + deltaT,
                               'endSimTime'             : lengthPrediction + lastEventDelta+deltaT,
                               'lastEventDelta'         : lastEventDelta,
                               'numEventsForList'       : numEventsForPrediction,
                               'gridWidth'              : gridWidth,
                               'gridHeight'             : gridHeight,
                               'eventTemp'              : eventTemp}
                    stochasticTimeLine[i], stochasticEventDict[i] = buildEventList(**kwargs)
                    eventDictForCalc = copy.deepcopy(stochasticEventDict[i])
                    eventDictForCalc.update(copy.deepcopy(filteredEvents))
                    timeLineForCalc = copy.deepcopy(stochasticTimeLine[i])
                    timeLineForCalc.append(continuesTimeLine[timeIndex])
                    timeLineForCalc.sort()
                    tempCost,tempEff,tempWaitTime = CalculateCostForKnownEventList(copy.deepcopy(carDictForCalc),copy.deepcopy(eventDictForCalc),timeLineForCalc,gridWidth,gridHeight)
                    if tempCost == 0:
                        tempCost = 0.001
                    stochasticCost.append(tempWaitTime+tempCost)
                # each set of events can happend with the same probability therefore this is a simple sum divided by num sets
                expectedCost = np.sum(stochasticCost)/len(stochasticCost)
                stochasticCost.append(expectedCost)
                costPerAction[j,:] = np.hstack(stochasticCost)
                # print('excpected value is:'+str(expectedCost))
                if costOfAction > expectedCost: # we want to minimize the expected cost of the plan since the cost is the movement of the cars and the time that each passanger waited
                    costOfAction = expectedCost
                    # print('excpected value chosen:'+str(expectedCost))
                    actionChosen = actions
            costPerActionList.append(costPerAction)
            for i,car in enumerate(carDict.values()):
                # if i == 0:
                    # print('actions Chosen:'+str(actionChosen))
                carDict[car['id']]['position'] = [a + b for a,b in zip(car['position'],actionChosen[i])]
            iterEnd = time.clock()
            timeIndex += 1
            print('time index: {0}, calculation time: {1}'.format(timeIndex, iterEnd-iterStart))
            print('total cost for this action is: ' +str(costOfAction))
    # dump logs
    with open('log_Cost_WaitTime_CarMovement_'+str(gridHeight)+'grid_'+str(numCars)+'cars_'+str(lengthSim)+'simLengh_'+str(numStochasticEvents)+'StochasticLength_'+str(lengthPrediction)+'Prediction'+'.p', 'wb') as out:
        pickle.dump({'cars': carPositionLog, 'events': eventDict, 'eventLog': eventLog}, out)

    # simulation summary
    plotingTimeline = continuesTimeLine[:-1]
    plt.figure(2)
    plt.plot(plotingTimeline, eventLog['count'], c='r', label='count')
    plt.plot(plotingTimeline, eventLog['closed'], c='b', label='closed')
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
    plt.legend()
    plt.xlim([-3, gridWidth + 3])
    plt.ylim([-3, gridHeight + 3])

    plt.show()
    return

if __name__=='__main__':
    main()
    print('Done.')