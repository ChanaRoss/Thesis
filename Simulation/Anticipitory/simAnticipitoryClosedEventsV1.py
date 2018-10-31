import numpy as np
from matplotlib import pyplot as plt
import copy,time,sys,pickle
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import imageio
from scipy.stats import truncnorm
import itertools as rt
import sys
sys.path.insert(0, '/home/chana/Documents/Thesis/FromGitFiles/SearchAlgorithm/')
from aStarClosedEvents_v1 import *
sns.set()


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



def unifromRandomEvents(startTime,endSimTime, n):
    """
    :param startTime: time to start the random list
    :param endSimTime: time to end the random list
    :param n: number of events
    :return: list of event times
    """
    return list(np.random.randint(startTime, endSimTime, n).astype(dtype=np.float32))

def poissonRandomEvents(startTime,endSimTime,lam):
    """
    creates time line assuming poisson distribution
    :param startTime: start time wanted for timeline
    :param endSimTime: end time wanted for timeline
    :param lam: lambda of poisson distribution (how often will an event appear?)
    :return: list of times for events
    """
    lengthTimeLine = int(endSimTime - startTime)
    numEventsPerTime = np.random.poisson(lam = lam,size = lengthTimeLine)
    eventTime = []
    for i in range(lengthTimeLine):
        nTime = numEventsPerTime[i]
        if nTime>0:
            for num in range(nTime):
                eventTime.append(i+startTime)
    return eventTime

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
    eventLog['answered'].append(len([e for e in eventDict.values() if e['answered']]))
    eventLog['canceled'].append(len([e for e in eventDict.values() if e['canceled']]))
    eventLog['current'].append(len(filterEvents(eventDict, currentTime)))
    eventLog['time'].append(currentTime)
    for event in eventDict.values():
        if event['timeStart'] <= currentTime and event['timeEnd']>currentTime :
            eventDict[event['id']]['statusLog'].append([currentTime, True])
        else:
            eventDict[event['id']]['statusLog'].append([currentTime, False])
        if event['timeEnd'] == currentTime and not event['answered']:
            eventDict[event['id']]['canceled'] = True
    return eventLog

def filterEvents(eventDict, currentTime):
    """
    :param eventDict: event dictionary at time Current time
    :param currentTime: the time wanted to be taken care of
    :return: dictionary of filtered events that are opened at time Current time
    """
    return {e['id']: e for e in eventDict.values() if not e['answered'] if e['timeStart'] <= currentTime if e['timeEnd']>currentTime}

def CalculateCostForKnownEventList(carDict,eventDict,weight,eventReward,eventPenalty):
    """
    :param carDict: car dictionary with all information for inner simulation
    :param eventDict:  event dictionary with all information for inner simulation
    :return: TotalCost (int): the total cost of this solution
            percentageEventsClosed (float): number of events answered/number of total events
            totalWaitTimeOfEvents(float): sum of time that each event was opened (if event was opened for n min.
            it will be sum(0,1,2,..n) for each event)
    """
    # main loop
    carsLoc         = np.vstack([c['position'] for c in carDict.values()])
    if len(eventDict)<1:
        print('no events to calculate')
        TotalCost = 0
    else:
        eventsLoc       = np.vstack([e['position'] for e in eventDict.values()])
        eventsTime      = np.array([e['timeStart'] for e in eventDict.values()])
        evnetsCloseTime = np.array([e['timeEnd'] for e in eventDict.values()])
        startTime       = np.min(eventsTime)
        eventsTime      = eventsTime - startTime
        evnetsCloseTime = evnetsCloseTime - startTime
        eventsAnswered  = np.zeros_like(eventsTime).astype(np.bool_)
        eventsCanceled  = np.zeros_like(eventsTime).astype(np.bool_)
        initState = SearchState(carsLoc, eventsLoc, eventsTime,evnetsCloseTime,eventReward,eventPenalty,eventsCanceled,eventsAnswered, float('inf'), 0, None,weight)
        p = aStar(initState)
        TotalCost = p[-1].gval
    return TotalCost

def buildEventList(startTime,endSimTime,timeWindow,lastEventDelta,lam,gridWidth,gridHeight,eventTemp):
    """
    :param startTime: time to start event list
    :param endSimTime: time to end event list
    :param timeWindow: fixed time for event to be opened
    :param lastEventDelta: last time to give an event
    :param lam: distribution wanted for events
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
    eventTimes = poissonRandomEvents(startTime,endSimTime-lastEventDelta,lam)
    eventPosX = truncnorm.rvs((0 - locX) / scaleX, (gridWidth - locX) / scaleX, loc=locX, scale=scaleX,
                              size=len(eventTimes)).astype(np.int64)
    eventPosY = truncnorm.rvs((0 - locY) / scaleY, (gridHeight - locY) / scaleY, loc=locY, scale=scaleY,
                              size=len(eventTimes)).astype(np.int64)

    # create event dict
    for i, et in enumerate(eventTimes):
        event = copy.deepcopy(eventTemp)
        event['timeStart'] = et
        event['timeEnd']   = timeWindow + et
        event['waitTime']  = event['timeEnd'] - event['timeStart']
        event['position']  = np.array([eventPosX[i], eventPosY[i]])
        event['id']        = i
        eventDict[i]       = event
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
    epsilon              = 0.1 # distance between locations to be considered same location
    numCars              = 2
    lam                  = 40/60 # number of events per hour/ 60
    lengthSim            = 35    # minutes
    gridWidth            = 8
    gridHeight           = 8
    timeWindow           = 3
    lastEventDelta       = 1
    numStochasticEvents  = 100
    eventReward          = 10
    canceledEventPenalty = 100
    openedEventPenalty   = 1
    astarWeight          = 1
    # initilize stochastic event list for each set checked -
    lengthPrediction     = 6
    # templates
    carEntityTemp = {'id': 0, 'velocity': 0, 'position': [0, 0], 'target': None, 'targetId': None, 'finished': 0}
    eventTemp = {'position': [], 'timeStart': 0, 'timeEnd': 0, 'answered': False,'canceled':False, 'id': 0, 'prob': 1, 'statusLog': [], 'waitTime': 0}

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
    eventLog = {'count': [], 'answered': [], 'canceled': [], 'current': [],'time':[]}
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
             'lam'                    : lam,
             'gridWidth'              : gridWidth,
             'timeWindow'             : timeWindow,
             'gridHeight'             : gridHeight,
             'eventTemp'              : eventTemp}
    timeLine, eventDict = buildEventList(**kwargs)
    # init simulation parameters -
    timeIndex = 0
    deltaT = 1
    continuesTimeLine = np.linspace(0, lengthSim+lastEventDelta, (lengthSim + lastEventDelta)/ deltaT + 1)
    totalCost = 0
    timeEachIndexTook = []
    while timeIndex < len(continuesTimeLine) - 1:
        iterStart = time.clock()
        costOfAction = 999999999
        actualCostOfAction = 999999999
        # find all events that are not closed and already started :
        filteredEvents = filterEvents(eventDict, continuesTimeLine[timeIndex])
        # log car positions and close events
        for car in carDict.values():
            tempFilteredEvents = copy.deepcopy(filteredEvents)
            if len(tempFilteredEvents) !=0:
                for event in tempFilteredEvents.values():
                    if np.linalg.norm(event['position'] - car['position'], 2) < epsilon:  # this car picked up this event
                        eventDict[event['id']]['answered'] = True
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
        # find all events that have been canceled in this iteration and add their cost to total cost:
        canceledEvents = len([e for e in eventDict.values() if e['timeEnd'] == continuesTimeLine[timeIndex] and not e['answered']])
        # penalty for all canceled events
        totalCost += canceledEvents * canceledEventPenalty
        for i in range(numStochasticEvents):
            # events should start after the events that are happening now
            kwargs = {'startTime'       : continuesTimeLine[timeIndex] + deltaT,
                      'endSimTime'      : lengthPrediction + deltaT + continuesTimeLine[timeIndex],
                      'lastEventDelta'  : lastEventDelta,
                      'lam'             : lam,
                      'gridWidth'       : gridWidth,
                      'gridHeight'      : gridHeight,
                      'timeWindow'      : timeWindow,
                      'eventTemp'       : eventTemp}
            stochasticTimeLine[i], stochasticEventDict[i] = buildEventList(**kwargs)
        costActionsList = []
        for j,actions in enumerate(rt.product(actionOptions.values(), repeat=len(carDict))):
            numEventsClosed    = 0
            # all costs that are effected by actions:
            eventsOpenedCost   = 0
            eventsAnsweredCost = 0
            actionCost = 0
            carDictForCalc     = {}
            FlagOutOfGrid      = False
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
                            # this event will be closed due to this action and therefore a reward is given
                            eventsAnsweredCost -= eventReward
                            numEventsClosed += 1 # number of closed events for this action
                        else: # the event is not answered by this action
                            eventsOpenedCost += openedEventPenalty
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
                if len(filteredEvents)>0:
                    eventDictForCalc.update(copy.deepcopy(filteredEvents))
                tempCost = CalculateCostForKnownEventList(carDict= copy.deepcopy(carDictForCalc),eventDict=copy.deepcopy(eventDictForCalc),weight=astarWeight,eventPenalty=canceledEventPenalty,eventReward=eventReward)
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
            print('event cost:' + str(eventsOpenedCost+eventsAnsweredCost))
            # print('num events opened:'+str(len(filteredEvents)))
            # print('num events closed:' + str(numEventsClosed))
            # print('num stochastic events:'+str([len(e) for e in stochasticEventDict.values()]))
            totalExpectedCost = expectedCost + actionCost+ eventsOpenedCost+eventsAnsweredCost
            costActionsList.append(totalExpectedCost)
            totalActualCost   = actionCost+ eventsOpenedCost+eventsAnsweredCost
            if costOfAction > totalExpectedCost: # we want to minimize the expected cost of the plan since the cost is the movement of the cars and the time that each passanger waited
                costOfAction = totalExpectedCost
                actualCostOfAction = totalActualCost
                # print('excpected value chosen:'+str(expectedCost))
                actionChosen = actions
                indexActionChosen = j
        for i,car in enumerate(carDict.values()):
            # if i == 0:
                # print('actions Chosen:'+str(actionChosen))
            carDict[car['id']]['position'] = [a + b for a,b in zip(car['position'],actionChosen[i])]
        iterEnd = time.clock()
        timeIndex += 1
        totalCost += actualCostOfAction
        timeEachIndexTook.append(iterEnd-iterStart)
        print('time index: {0}, calculation time: {1}'.format(timeIndex, iterEnd-iterStart))
        print('total cost for this action is: ' +str(costOfAction))
        print(costActionsList)
        print('action chosen is:' + str(actionChosen))
        print('index of chosen action is:'+str(indexActionChosen))
        # dump logs
        with open('logAnticipatory_'+str(eventReward)+'EventReward_'+str(gridHeight)+'grid_'+str(numCars)+'cars_'+str(lengthSim)+'simLengh_'+str(numStochasticEvents)+'StochasticLength_'+str(lengthPrediction)+'Prediction_'+str(astarWeight)+'aStarWeight.p', 'wb') as out:
            pickle.dump({'cars'              : carPositionLog,
                         'events'            : eventDict,
                         'eventLog'          : eventLog,
                         'carDict'           : carDict,
                         'gridSize'          : gridHeight,
                         'simLength'         : lengthSim,
                         'cost'              : totalCost,
                         'numStochasticRuns' : numStochasticEvents,
                         'TimeEachStepTook'  : timeEachIndexTook,
                         'lengthPrediction'  : lengthPrediction}, out)

            # simulation summary
    plotingTimeline = continuesTimeLine[:-1]
    plt.figure(2)
    plt.plot(plotingTimeline, eventLog['count'], c='r', label='count')
    plt.plot(plotingTimeline, eventLog['answered'], c='b', label='answered')
    plt.plot(plotingTimeline, eventLog['canceled'], c='y',label = 'canceled')
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
        if event['answered']:
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