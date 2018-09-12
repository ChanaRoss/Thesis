import numpy as np
from matplotlib import pyplot as plt
import copy,time,sys,pickle
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import imageio
from scipy.stats import truncnorm
import itertools as rt
import sys
sys.path.insert(0, '/home/chana/Documents/Thesis/SearchAlgorithm/')
from aStar_v4 import *
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
    """
    :param startTime: time to start the random list
    :param endSimTime: time to end the random list
    :param n: number of events
    :return: list of event times
    """
    return list(np.random.randint(startTime, endSimTime, n).astype(dtype=np.float32))

def createCarPositionLog(car, currentTime):
    """
    :param car: dictionary with spesific car parameters at time Current time
    :param currentTime: the time wanted to be logged
    :return: dictionary to add to car logger
    """
    return {'position': car['position'], 'time': currentTime, 'target': car['target'], 'targetId': car['targetId']}

def createCarUseLog(carDict, currentTime):
    """
    :param carDict: dictionary of cars
    :param currentTime: time wanted to be logged
    :return: number of cars in use and not in use at time Current time
    """
    inUse = len([c for c in carDict.values() if c['target'] is not None])
    notInUse = len(carDict)-inUse
    return {'occupied': inUse, 'idle': notInUse}

def createEventLog(eventLog, eventDict, currentTime):
    """
    :param eventLog: event log of number of events closed,created and canceled
    :param eventDict: dictionary of events
    :param currentTime: time wanted to be logged
    :return: updated event log and status log in event dictionary
    """
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
    """
    :param eventDict: event dictionary at time Current time
    :param currentTime: the time wanted to be taken care of
    :return: dictionary of filtered events that are opened at time Current time
    """
    return {e['id']: e for e in eventDict.values() if not e['closed'] if e['timeStart'] <= currentTime if e['timeEnd']>currentTime}

def CalculateCostForKnownEventList(carDict,eventDict,timeLine,gridWidth,gridHeight,weight):
    """
    :param carDict: car dictionary with all information for inner simulation
    :param eventDict:  event dictionary with all information for inner simulation
    :param timeLine: time line for inner simulation
    :param gridWidth: grid size [m]
    :param gridHeight: grid size [m], assume it is the same as gridWidth
    :return: TotalCost (int): the total cost of this solution
            percentageEventsClosed (float): number of events answered/number of total events
            totalWaitTimeOfEvents(float): sum of time that each event was opened (if event was opened for n min.
            it will be sum(0,1,2,..n) for each event)
    """
    # main loop
    numCars = len(carDict)
    carsLoc      = np.vstack([c['position'] for c in carDict.values()])
    eventsLoc    = np.vstack([e['position'] for e in eventDict.values()])
    eventsTime   = np.array([e['timeStart'] for e in eventDict.values()])
    startTime    = np.min(eventsTime)
    eventsTime   = eventsTime - startTime
    eventsStatus = np.ones_like(eventsTime).astype(np.bool_)
    initState = SearchState(carsLoc, eventsLoc, eventsTime, eventsStatus, float('inf'), 0, None,weight)
    p = aStar(initState)
    TotalCost = p[-1].gval
    return TotalCost

def buildEventList(startTime,endSimTime,lastEventDelta,numEventsForList,gridWidth,gridHeight,eventTemp):
    """
    :param startTime: time to start event list
    :param endSimTime: time to end event list
    :param lastEventDelta: last time to give an event
    :param numEventsForList: number of events wanted
    :param gridWidth: grid size for position of events
    :param gridHeight: grid size for position of events
    :param eventTemp: the template for the event dictionary
    :return: timeLine (list) : the time line for the events
             eventDict (dict) : the dictionary of events created (time, location, etc.)
    """
    # init event dict
    eventDict = {}
    locX = gridWidth / 2
    scaleX = gridWidth/3
    locY = gridHeight / 2
    scaleY = gridHeight/3
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
        event['position'] = np.array([eventPosX[i], eventPosY[i]])
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
    numEvents = 40
    lengthSim = 40 # minutes
    gridWidth = 7
    gridHeight = 7
    eventDuration = lengthSim
    lastEventDelta = 1
    numStochasticEvents = 100
    astarWeight = 3
    # templates
    carEntityTemp = {'id': 0, 'velocity': 0, 'position': [0, 0], 'target': None, 'targetId': None, 'finished': 0}
    eventTemp = {'position': [], 'timeStart': 0, 'timeEnd': 0, 'closed': False, 'id': 0, 'prob': 1, 'statusLog': [], 'waitTime': 0}

    # initialize action dictionary - assuming at each deltaTime one step is taken (can change based on velocity of each car
    actionOptions = {'Up': np.array([0, 1]), 'Down': np.array([0, -1]), 'Left': np.array([-1, 0]), 'Right': np.array([1, 0]),'stay':np.array([0,0])}

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
        car['position'] = np.array([int(np.random.randint(0, gridWidth, 1)), int(np.random.randint(0, gridHeight, 1))])
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
    lengthPrediction = 3
    # number of events for prediction is the number of events in simulation divided by the simulation time, for same amount per hour
    numEventsForPrediction = np.max([lengthPrediction*(numEvents//lengthSim),1])
    deltaT = 1
    continuesTimeLine = np.linspace(0, lengthSim+lastEventDelta, (lengthSim + lastEventDelta)/ deltaT + 1)
    totalCost = 0
    timeEachIndexTook = []
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
                    if np.linalg.norm(event['position'] - car['position'], 2) < epsilon:  # this car picked up this event
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

        numOptions = 5**(len(carDict))
        for i in range(numStochasticEvents):
            # events should start after the events that are happening now
            kwargs = {'startTime'       : continuesTimeLine[timeIndex] + deltaT,
                      'endSimTime'      : lengthPrediction + deltaT + continuesTimeLine[timeIndex],
                      'lastEventDelta'  : lastEventDelta,
                      'numEventsForList': numEventsForPrediction,
                      'gridWidth'       : gridWidth,
                      'gridHeight'      : gridHeight,
                      'eventTemp'       : eventTemp}
            stochasticTimeLine[i], stochasticEventDict[i] = buildEventList(**kwargs)
        for j,actions in enumerate(rt.product(actionOptions.values(), repeat=len(carDict))):
            carDictForCalc = {}
            FlagOutOfGrid = False
            actionCost = 0
            if len(filteredEvents)>0:
                eventTimeCost = len(filteredEvents)
            else:
                eventTimeCost = 0 # no events opened at this time step
            # create dictionary of cars for the stochastic calculations to find the cost of each action combination
            for i,car in enumerate(carDict.values()):
                carWithActioin = car.copy()
                carWithActioin['position'] =carWithActioin['position'] + actions[i]
                ActionDist = manhattenDist(carWithActioin['position'],car['position'])
                actionCost += ActionDist
                carDictForCalc[car['id']] = carWithActioin
                if len(filteredEvents)>0:
                    for event in filteredEvents.values():
                        if manhattenDist(carWithActioin['position'],event['position'])<epsilon:
                            eventTimeCost -=1 # this event will be closed due to this action and therfore wont cost anything
                if np.abs(carWithActioin['position'][0])>gridWidth or np.abs(carWithActioin['position'][1])>gridHeight:
                    FlagOutOfGrid = True
                    carIndexOutOfGrid = car['id']
            if FlagOutOfGrid:
                print('action out of bounds for car:'+str(carIndexOutOfGrid))
                continue

            # calculate the cost of n eventLists for each action combination to find the expected value of moving in these actions
            stochasticCost = []
            timeStoch = []
            for i in range(numStochasticEvents):
                startTimeStoch  = time.clock()
                eventDictForCalc = copy.deepcopy(stochasticEventDict[i])
                timeLineForCalc = copy.deepcopy(stochasticTimeLine[i])
                if len(filteredEvents)>0:
                    eventDictForCalc.update(copy.deepcopy(filteredEvents))
                    timeLineForCalc.append(continuesTimeLine[timeIndex])
                timeLineForCalc.sort()
                tempCost = CalculateCostForKnownEventList(copy.deepcopy(carDictForCalc),copy.deepcopy(eventDictForCalc),timeLineForCalc,gridWidth,gridHeight,weight=astarWeight)
                if tempCost == 0:
                    tempCost = 0.001
                stochasticCost.append(tempCost)
                endTimeStoch = time.clock()
                timeItter = round(endTimeStoch - startTimeStoch,3)
                timeStoch.append(timeItter)
                # print('time iteration number:'+str(i)+' took: ' + str(timeItter))
            # each set of events can happend with the same probability therefore this is a simple sum divided by num sets
            expectedCost = np.sum(stochasticCost)/len(stochasticCost)
            print('action check number:' + str(j))
            print('total time stochastic runs took' + str(np.sum(timeStoch)))
            print('stochastic cost:'+str(stochasticCost))
            print('expected cost:' + str(expectedCost))
            print('action cost:' +str(actionCost))
            print('event cost:' + str(eventTimeCost))
            print('num events opened:'+str(len(filteredEvents)))
            print('num stochastic events:'+str([len(e) for e in stochasticEventDict.values()]))
            # print('excpected value is:'+str(expectedCost))
            if costOfAction > expectedCost + actionCost+ eventTimeCost: # we want to minimize the expected cost of the plan since the cost is the movement of the cars and the time that each passanger waited
                costOfAction = expectedCost + actionCost + eventTimeCost
                # print('excpected value chosen:'+str(expectedCost))
                actionChosen = actions
        for i,car in enumerate(carDict.values()):
            # if i == 0:
                # print('actions Chosen:'+str(actionChosen))
            carDict[car['id']]['position'] = [a + b for a,b in zip(car['position'],actionChosen[i])]
        iterEnd = time.clock()
        timeIndex += 1
        totalCost += costOfAction
        timeEachIndexTook.append(iterEnd-iterStart)
        print('time index: {0}, calculation time: {1}'.format(timeIndex, iterEnd-iterStart))
        print('total cost for this action is: ' +str(costOfAction))
        print('action chosen is:' + str(actionChosen))
        # dump logs
        with open('log_Cost_WaitTime_CarMovement_'+str(gridHeight)+'grid_'+str(numCars)+'cars_'+str(lengthSim)+'simLengh_'+str(numStochasticEvents)+'StochasticLength_'+str(lengthPrediction)+'Prediction_'+str(astarWeight)+'aStarWeight.p', 'wb') as out:
            pickle.dump({'cars'              : carPositionLog,
                         'events'            : eventDict,
                         'eventLog'          : eventLog,
                         'carDict'           : carDict,
                         'gridSize'          : gridHeight,
                         'simLength'         : lengthSim,
                         'numStochasticRuns' : numStochasticEvents,
                         'TimeEachStepTook'  : timeEachIndexTook,
                         'lengthPrediction'  : lengthPrediction}, out)

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