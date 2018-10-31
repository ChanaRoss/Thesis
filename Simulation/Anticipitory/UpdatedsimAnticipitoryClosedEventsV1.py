# for stats on running time
import copy,time,sys,pickle
# for mathematical calculations and statistical distributions
from scipy.stats import truncnorm
from scipy.spatial.distance import cdist
from scipy.special import comb
import math
import itertools
import numpy as np
# my files
sys.path.insert(0, '/home/chana/Documents/Thesis/FromGitFiles/SearchAlgorithm/')
from aStarClosedEvents_v1 import *
# for graphics
import seaborn as sns
from matplotlib import pyplot as plt
sns.set()


class Logs():
    def __init__(self,numEvents,numCars,simLength):
        self.eventsAnswered = np.zeros(shape = (simLength,1))
        self.eventsCanceled = np.zeros_like(self.eventsAnswered)
        self.eventsCommited = np.zeros_like(self.eventsAnswered)
        self.eventsOpened   = np.zeros_like(self.eventsAnswered)
        self.eventsCreated  = np.zeros_like(self.eventsAnswered)
        self.timeVector     = np.zeros_like(self.eventsAnswered)

    def UpateLogs(self,timeVector,timeIndex,events):
        self.eventsAnswered[timeIndex,:] = copy.deepcopy(np.sum(events.answered))
        self.eventsCanceled[timeIndex,:] = copy.deepcopy(np.sum(events.canceled))
        self.eventsCommited[timeIndex,:] = copy.deepcopy(np.sum(events.committed))
        self.eventsOpened[timeIndex,:]   = copy.deepcopy(np.sum(events.openedEvents))
        self.eventsCreated[timeIndex,:]  = copy.deepcopy(np.sum(events.timeWindows[:,0]<timeVector[timeIndex]))
        self.timeVector[timeIndex]       = timeVector[timeIndex]

class Events():
    def __init__(self,eventsPos,eventsTimeWindow):
        self.positions    = eventsPos
        self.timeWindows  = eventsTimeWindow
        self.numEvents    = eventsPos.shape[0]
        self.answered     = np.zeros(shape=(self.numEvents,1)).astype(np.bool_)
        self.canceled     = np.zeros(shape=(self.numEvents,1)).astype(np.bool_)
        self.openedEvents = np.zeros(shape=(self.numEvents,1)).astype(np.bool_)
        # events commited to , these events will be answered no matter what, even if time window is closed
        self.committed      = np.zeros(shape=(self.numEvents,1)).astype(np.bool_)
        self.timeAnswered   = np.zeros(shape=(self.numEvents,1))
        self.index          = np.array(range(self.numEvents)).reshape(self.numEvents,1)

    def UpdateOpenedEvents(self,t):
        """
        update opened events status
        :param t: current time in simulation , relevant for time window
        :return:
        """
        # find events not answered or committed to
        step1 = np.logical_and(np.logical_not(self.committed),np.logical_not(self.answered))
        openedEvents = np.logical_and(step1,np.logical_not(self.canceled))
        relevantTimeWindows = np.logical_and(self.timeWindows[:,0]<=t,self.timeWindows[:,1]>t)
        self.openedEvents = np.logical_and(openedEvents.flatten(),relevantTimeWindows)

    def UpdateCanceledEvents(self,t):
        outOfTimeBool = self.timeWindows[:,1] < t
        currentlyCanceled = np.logical_and(outOfTimeBool.reshape(self.committed.shape),np.logical_not(self.committed))
        self.canceled[currentlyCanceled] = 1
        return np.sum(currentlyCanceled)

    def GetNumOpenedEvents(self):
        return np.sum(self.openedEvents)

    def GetNumCanceledEvents(self):
        return np.sum(self.canceled)

    def GetOpenedEvents(self):
        return self.openedEvents



class Cars():
    def __init__(self,initPos):
        self.poistions        = initPos
        self.numCars          = self.poistions.shape[0]
        self.committed        = np.zeros(shape=(self.numCars,1)).astype(np.bool_)
        self.commitEventIndex = np.zeros(shape=(self.numCars,1)).astype(np.bool_)
        self.index            = np.array(range(self.numCars)).reshape(self.numCars,1)
        self.path             = [self.poistions]
    def UpdateCarsPosition(self,actionVector):
        self.poistions += actionVector

    def GenNumberOfAvailabelCars(self):
        return np.sum(self.committed)


# Function definitions

def GetFilteredCars(cars, filteredIndex):
    """
    returns a class of filtered cars by index's wanted
    :param cars: class of cars in current time
    :param filteredIndex: indexs of cars wanted
    :return: class of filtered cars
    """
    carPos =  copy.deepcopy(cars.poistions[filteredIndex])
    tempCars             = Cars(carPos)
    tempCars.index       = filteredIndex
    tempCars.committed   = copy.deepcopy(cars.committed[filteredIndex])
    tempCars.commitEventIndex = copy.deepcopy(cars.commitEventIndex[filteredIndex])
    return tempCars

def GetFilteredEvents(events, filteredIndex):
    """
    returns a class of filered events by index's wanted
    :param events: class of events in current time
    :param filteredIndex: indexs of events watned
    :return: class of filtered events
    """
    eventsPos           = copy.deepcopy(events.positions[filteredIndex])
    eventsTimes         =  copy.deepcopy(events.timeWindows[filteredIndex, :])
    tempEvents          = Events(eventsPos, eventsTimes)
    tempEvents.index    = filteredIndex
    tempEvents.answered = copy.deepcopy(events.answered[filteredIndex])
    tempEvents.committed= copy.deepcopy(events.committed[filteredIndex])
    tempEvents.canceled = copy.deepcopy(events.canceled[filteredIndex])
    tempEvents.openedEvents = copy.deepcopy(events.openedEvents[filteredIndex])
    tempEvents.timeAnswered = copy.deepcopy(events.timeAnswered[filteredIndex])
    return tempEvents

def createMoveMatrix(numCars):
    """
    returns a 3D array:
    mat[i][j][0] = delta of car j in possible move i in x axis
    mat[i][j][1] = delta of car j in possible move i in y axis
    :param numCars: number of cars in scenario
    :return: numpy integer array (int8)
    """
    return np.array(list(itertools.product([(0,0), (0,1), (0,-1), (-1,0), (1,0)], repeat=numCars))).astype(np.int8)


def calcNewCarPos(carsPos, moveMat):
    """
    calculate all the new car position possibilities
    :param carsPos: car positions at the current time step
    :param moveMat: 3D matrix of possible moves shape=(num options,nc,2)
    :return: 3D matrix of all possible positions, shape=(num options,nc,2) (int32)
    """
    tiledCarPos = np.tile(carsPos, reps=(moveMat.shape[0], 1, 1))
    newCarPos = tiledCarPos + moveMat
    return newCarPos.astype(np.int32)


def calculateCostOfAction(distanceMatrix,moveMat,eventsReward , openedEventPenalty,epsilon = 0.01):
    moveCost             = np.sum(np.absolute(moveMat),axis = (1,2))
    if distanceMatrix is None:
        # count how many events are reached in current time assuming spesific action
        numAnsweredEvents = 0
        eventsAnsweredReward = numAnsweredEvents * eventsReward
        numEventsOpened = 0
    else:
        # count how many events are reached in current time assuming spesific action
        numAnsweredEvents    = np.sum(np.sum((distanceMatrix <= epsilon), axis=1) >= 1,axis = 1)
        eventsAnsweredReward = numAnsweredEvents*eventsReward
        numEventsOpened      = distanceMatrix.shape[2] - numAnsweredEvents
    return moveCost - eventsAnsweredReward + numEventsOpened*openedEventPenalty



def calcDistanceMatrix(possibleCarPos, eventPos):
    distanceStack = np.zeros(shape=(possibleCarPos.shape[0], possibleCarPos.shape[1], eventPos.shape[0]))
    for i in range(possibleCarPos.shape[0]):
        distanceStack[i, :, :] = cdist(possibleCarPos[i,:,:], eventPos, metric='cityblock')
    return distanceStack.astype(np.int32)



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
    return np.array(eventTime)

def createEventsDistribution(gridWidth, gridHeight, startTime, endTime, lam, eventTimeWindow):
    locX        = gridWidth / 2
    scaleX      = gridWidth / 3
    locY        = gridHeight / 2
    scaleY      = gridHeight / 3
    # randomize event times
    eventTimes  = poissonRandomEvents(startTime, endTime, lam)
    eventPosX   = truncnorm.rvs((0 - locX) / scaleX, (gridWidth - locX) / scaleX, loc=locX, scale=scaleX,
                              size=len(eventTimes)).astype(np.int64)
    eventPosY   = truncnorm.rvs((0 - locY) / scaleY, (gridHeight - locY) / scaleY, loc=locY, scale=scaleY,
                              size=len(eventTimes)).astype(np.int64)

    eventsPos           = np.column_stack([eventPosX,eventPosY])
    eventsTimeWindow    = np.column_stack([eventTimes,eventTimes+eventTimeWindow])
    return eventsPos,eventsTimeWindow


def createCommitMatrix(numCars,numEvents):
    if numEvents == 0:
        commitMat = np.zeros(shape=(numCars,1)).astype(np.bool_)
    else:
        numCases = 0
        numOptionalCommits = np.min([numCars+1,numEvents+1])
        numCasesPerCommitOptions = []
        for i in range(np.min([numCars+1,numEvents+1])):
            numCasesPerCommitOptions.append(comb(numCars,i)*comb(numEvents,i)*math.factorial(i))

        numCases = np.sum(numCasesPerCommitOptions)
        commitMat = np.zeros(shape = (int(numCases),numCars,numEvents))
        k = 0
        for i in range(numOptionalCommits):
            for carChoice in list(itertools.combinations(range(numCars),i)):
                for eventChoice in list(itertools.combinations(range(numEvents),i)):
                    for eventChoicePermutations in list(itertools.permutations(eventChoice,len(eventChoice))):
                        commitMat[k,carChoice,eventChoicePermutations] = True
                        k += 1
    return commitMat

def updateEventsAnsweredStatus(cars,events,time,epsilon= 0.01):
    eventsPos = events.positions
    carsPos   = cars.poistions
    distanceMatrix = cdist(carsPos, eventsPos, metric='cityblock')
    eventsOpened   = events.openedEvents
    # convert distance matrix to boolean of approx zero (picked up events)
    step1          = np.sum((distanceMatrix <= epsilon), axis=0) >= 1
    # condition on event being open
    step2          = np.logical_and(step1, eventsOpened)
    eventsAnswered = copy.deepcopy(events.answered)
    # new possible events answered status
    eventsAnswered[step2] = 1
    eventsAnswered.astype(np.bool_)
    eventsTimeAnswered = copy.deepcopy(events.timeAnswered)
    eventsTimeAnswered[step2] = time
    return eventsAnswered,eventsTimeAnswered

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

def updateCommitedCarsPos(cars,events,commitedCarIndex,commitedEventIndex,epsilon=0.01):
    possibleDistance = 1
    for eventIndex,carIndex in zip(commitedEventIndex,commitedCarIndex):
        if events.canceled[eventIndex]:
            # event was canceled before car could reach event. car becomes available again
            cars.commitEventIndex[carIndex] = np.nan
            cars.committed[carIndex]        = False
        else:
            dx, dy = manhattenPath(cars.poistions[carIndex], events.positions[eventIndex])
            if np.random.binomial(1, 0.5):
                ix = np.min([possibleDistance, abs(dx)]) * np.sign(dx)
                iy = np.max([possibleDistance - abs(ix), 0]) * np.sign(dy)
            else:
                iy = np.min([possibleDistance, abs(dy)]) * np.sign(dy)
                ix = np.max([possibleDistance - abs(iy), 0]) * np.sign(dx)
            # update car location towards event
            cars.poistions[carIndex] += np.array([ix, iy])


def createStochasticEvents(numStochasticRuns, notCommitedEvents,gridWidth,gridHeight,startTime,endTime,lam,eventsTimeWindow):
    """
    create list of stochastic event sets based on the same distribution function as the original problem
    :param numStochasticRuns: number of sets to produce
    :param notCommitedEvents: the actual events currently opened
    :param gridWidth: grid size
    :param gridHeight: grid size
    :param startTime: time at which to produce the events
    :param endTime: closing time for events based on predection wanted
    :param lam: lamda of poission ditribution
    :param eventsTimeWindow: time window defined for event (amount of time event is alive)
    :return: list of sets, each containing a class of events to run a* on
    """
    stochasticEventsList = []
    if notCommitedEvents is not None:
        realPositions = copy.deepcopy(notCommitedEvents.positions)
        realTimes     = copy.deepcopy(notCommitedEvents.timeWindows)
    for i in range(numStochasticRuns):
        eventPos,eventTimeWindow = createEventsDistribution(gridWidth, gridHeight, startTime, endTime, lam, eventsTimeWindow)
        if notCommitedEvents is None:
            stochasticEventPos    = eventPos
            stochasticTimeWindows = eventTimeWindow
        else:
            stochasticEventPos       = np.concatenate((realPositions,eventPos))
            stochasticTimeWindows    = np.concatenate((realTimes,eventsTimeWindow))
        if stochasticTimeWindows.size >0:
            stochasticTimeWindows    = stochasticTimeWindows - np.min(stochasticTimeWindows[0,:]) # start time vector in stochastic events from 0
        stochasticEventsList.append(Events(stochasticEventPos,stochasticTimeWindows))
    return stochasticEventsList

def main():
    # define starting positions in the simulation:
    # set seed
    seed = 1
    np.random.seed(seed)
    shouldPrint          = True
    # params
    epsilon              = 0.1 # distance between locations to be considered same location
    numCars              = 2
    lam                  = 10/30 # number of events per hour/ 60
    lengthSim            = 35    # minutes
    gridWidth            = 8
    gridHeight           = 8
    timeWindow           = 3
    addedComittedTime    = 5
    numStochasticRuns    = 2
    eventReward          = 10
    canceledEventPenalty = 100
    openedEventPenalty   = 1
    astarWeight          = 1
    # initilize stochastic event list for each set checked -
    lengthPrediction     = 6
    deltaT               = 1

    continuesTimeLine   = np.linspace(0, lengthSim, lengthSim/ deltaT + 1)
    totalCost           = 0
    timeIndex           = 0
    timeEachIndexTook   = []

    # create initial distributions of car and event data:
    carsPositions       = np.zeros(shape=(numCars,2))
    for i in range(numCars):
        carsPositions[i,:] = [int(np.random.randint(0, gridWidth, 1)), int(np.random.randint(0, gridHeight, 1))]
    eventsPos,eventsTimeWindow = createEventsDistribution(gridWidth, gridHeight, 0, lengthSim, lam, timeWindow)
    # initialize cars and events classes with distributions found:
    cars            = Cars(carsPositions)
    events          = Events(eventsPos,eventsTimeWindow)
    events.UpdateOpenedEvents(t=0) # update vector for initial time
    events.answered,events.timeAnswered = updateEventsAnsweredStatus(cars,events,0)
    while timeIndex < len(continuesTimeLine) - 1:
        # update opened events list and canceled events :
        events.UpdateOpenedEvents  (t = continuesTimeLine[timeIndex])
        numCanceled  = events.UpdateCanceledEvents(t = continuesTimeLine[timeIndex])
        # add to cost penalty for currently canceled events -
        totalCost       = totalCost + numCanceled*canceledEventPenalty
        numOpenedEvents = np.sum(events.openedEvents)
        if np.sum(cars.committed):
            # there are cars that are committed to events and need to move towards them one step (only moves if event is still opened)
            commitedCarIndex = cars.index[cars.committed]
            # find positions of events that the cars are committed to
            commitedEventIndex = events.index[cars.commitEventIndex[np.logical_not(np.isnan(cars.commitEventIndex))]]
            updateCommitedCarsPos(cars, events,commitedCarIndex,commitedEventIndex)
        # find relevant cars that are not commited (the number of committed cars + number of available cars should be the number of cars
        relevantCarsIndex  = cars.index[np.logical_not(cars.committed)]
        numRelevantCars = np.sum(np.logical_not(cars.committed))
        if numRelevantCars>0:
            if np.sum(events.openedEvents):
                # there are opened events and therefore should choose if cars need to commit to opened events and also if cars should move one step
                relevantEventIndex = events.index[events.openedEvents]
                commitMat          = createCommitMatrix(numRelevantCars,numOpenedEvents)
                isEventCommited    = np.sum(commitMat, axis=1).astype(np.bool_)
                numCommitedEvents  = np.sum(isEventCommited, axis=1)
                # these will be the indexs of chosed value in commit and move matrix
                optionalMoveIndex        = np.empty(shape=(commitMat.shape[0], 1))
                optionalMoveCost         = np.zeros_like(optionalMoveIndex)
                optionalRealCost         = np.zeros_like(optionalMoveIndex)
                optionalCommitCost       = np.zeros_like(optionalMoveIndex)
                optionalMoveIndex[:]     = np.nan
                optionalMoveCost[:]      = np.nan
                optionalRealCost[:]      = np.nan

                isCarCommited            = np.sum(commitMat, axis=2).astype(np.bool_)
                optionalCommitCost       = optionalCommitCost - numCommitedEvents * eventReward
                numNotCommitedCars       = np.sum(np.logical_not(isCarCommited), axis=1)
                numCommitedCars          = np.sum(isCarCommited,axis = 1)
                for i in range(commitMat.shape[0]):
                    if numNotCommitedCars[i]:
                        # there are cars that are not commited and need to be treated using move car options
                        moveMat              = createMoveMatrix(numNotCommitedCars[i])
                        if np.sum(np.logical_not(isEventCommited)):
                            # there are no opened events that are not commited
                            notCommitedEvents = None
                            optionalActionCost = calculateCostOfAction(None, moveMat, eventReward, openedEventPenalty, epsilon=0.01)
                        else:
                            notCommitedEventIndex = relevantEventIndex[np.logical_not(isEventCommited[i,:])]
                            notCommitedEvents     = GetFilteredEvents(events, notCommitedEventIndex)
                            optionalNewCarsPos = calcNewCarPos(copy.deepcopy(notCommitedCars.poistions), moveMat)

                            distanceMatrix     = calcDistanceMatrix(optionalNewCarsPos, notCommitedEvents.positions)
                            optionalActionCost = calculateCostOfAction(distanceMatrix, moveMat, eventReward,openedEventPenalty, epsilon=0.01)
                        optionalExpectedCost = np.zeros(shape=(moveMat.shape[0], 1))
                        notCommitedCarsIndex = relevantCarsIndex[np.logical_not(isCarCommited[i,:])]
                        notCommitedCars      = GetFilteredCars(cars,notCommitedCarsIndex)
                        stochasticEventsList = createStochasticEvents(numStochasticRuns, notCommitedEvents, gridWidth,
                                                                      gridHeight,
                                                                      continuesTimeLine[timeIndex] + 1,
                                                                      continuesTimeLine[timeIndex + 1] + lengthPrediction, lam,
                                                                      timeWindow)
                        for j in range(moveMat.shape[0]):  # check each move and see if it was worth while
                            newCarPos = optionalNewCarsPos[j, :, :]
                            stochasticCost = np.zeros(shape=(numStochasticRuns, 1))
                            # running stochastic runs in a* to find optimal solution for deterministic problem
                            for numRun, stochasticEvent in enumerate(stochasticEventsList):
                                eventsPos = copy.deepcopy(stochasticEvent.positions)
                                eventsTime = copy.deepcopy(stochasticEvent.timeWindows[:, 0])
                                eventsCloseTime = copy.deepcopy(stochasticEvent.timeWindows[:, 1])
                                eventsCanceled = np.zeros_like(eventsTime).astype(np.bool_)
                                eventsAnswered = np.zeros_like(eventsTime).astype(np.bool_)
                                initState = SearchState(newCarPos, eventsPos, eventsTime, eventsCloseTime, eventReward,
                                                        canceledEventPenalty,
                                                        eventsCanceled, eventsAnswered, float('inf'), 0, None, astarWeight)
                                p = aStar(initState)
                                stochasticCost[numRun] = p[-1].gval
                                if shouldPrint:
                                    print('num stochastic run is: '+str(numRun) +'/'+str(len(stochasticEventsList)))
                                    print('a star results is:'+str(stochasticCost[numRun]))
                            optionalExpectedCost[j] = np.mean(stochasticCost)
                            if shouldPrint:
                                print('num Checked move is: '+str(j)+'/'+str(moveMat.shape[0]))
                                print('expected cost is: '+str(optionalExpectedCost[j]))
                        optionalActionCost    = optionalActionCost.reshape(optionalExpectedCost.shape)
                        optionalTotalCost     = optionalActionCost + optionalExpectedCost + totalCost
                        optionalMoveIndex[i]  = np.argmin(optionalTotalCost).astype(int)
                        optionalMoveCost[i]   = np.min(optionalTotalCost)
                        optionalRealCost[i]   = optionalActionCost[optionalMoveIndex[i].astype(int)]
                        optionalCommitCost[i] = optionalCommitCost[i] + optionalMoveCost[i]
                    else:
                        optionalActionCost    = np.sum(numCommitedCars)
                        optionalTotalCost     = optionalActionCost + totalCost
                        optionalMoveIndex[i]  = 0
                        optionalMoveCost[i]   = np.min(optionalTotalCost)
                        optionalRealCost[i]   = optionalActionCost[optionalMoveIndex[i]]
                        optionalCommitCost[i] = optionalCommitCost[i] + optionalMoveCost[i]

                chosenCommitIndex = np.argmin(optionalCommitCost)
                chosenCost        = np.min(optionalCommitCost)
                chosenRealCost    = optionalRealCost[chosenCommitIndex]
                chosenMoveIndex   = optionalMoveIndex[chosenCommitIndex]
                # make changes for chosen action from all options of commits and movements :
                if np.sum(isEventCommited):
                    chosenEventsCommitedIndex = relevantEventIndex[isEventCommited[chosenCommitIndex, :]]
                    chosenCarsCommitedIndex   = relevantCarsIndex[isCarCommited[chosenCommitIndex, :]]
                    # update cars that are commited to True and events that are commited to True
                    cars.committed[chosenCarsCommitedIndex]     = True
                    chosenCommit = commitMat[chosenCommitIndex]
                    eventIndex = np.argmax(chosenCommit,axis = 1)
                    cars.commitEventIndex[chosenCarsCommitedIndex] = chosenEventsCommitedIndex
                    events.committed[chosenEventsCommitedIndex] = True
                    # add time to comitted events since now they are worth more
                    events.timeWindows[chosenEventsCommitedIndex, 1] += addedComittedTime
                    updateCommitedCarsPos(cars, events, chosenCarsCommitedIndex, chosenEventsCommitedIndex)
                moveMat = createMoveMatrix(numNotCommitedCars[chosenCommitIndex])
                chosenCarMoves = moveMat[chosenMoveIndex, :, :]
                cars.poistions[chosenCarsCommitedIndex] = cars.poistions[chosenCarsCommitedIndex] + chosenCarMoves

            else:
                moveMat = createMoveMatrix(numRelevantCars)
                relevantCars = GetFilteredCars(cars, relevantCarsIndex)
                stochasticEventsList = createStochasticEvents(numStochasticRuns, None, gridWidth,
                                                              gridHeight,
                                                              continuesTimeLine[timeIndex] + 1,
                                                              continuesTimeLine[timeIndex + 1] + lengthPrediction, lam,
                                                              timeWindow)
                optionalNewCarsPos   = calcNewCarPos(copy.deepcopy(relevantCars.poistions), moveMat)
                optionalExpectedCost = np.zeros(shape=(moveMat.shape[0]))
                optionalActionCost   = calculateCostOfAction(None, moveMat, eventReward, openedEventPenalty, epsilon=0.01)
                for j in range(moveMat.shape[0]):  # check each move and see if it was worth while
                    newCarPos = optionalNewCarsPos[j, :, :]
                    stochasticCost = np.zeros(shape=(numStochasticRuns))
                    # running stochastic runs in a* to find optimal solution for deterministic problem
                    for numRun, stochasticEvent in enumerate(stochasticEventsList):
                        eventsPos = copy.deepcopy(stochasticEvent.positions)
                        eventsTime = copy.deepcopy(stochasticEvent.timeWindows[:, 0])
                        eventsCloseTime = copy.deepcopy(stochasticEvent.timeWindows[:, 1])
                        eventsCanceled = np.zeros_like(eventsTime).astype(np.bool_)
                        eventsAnswered = np.zeros_like(eventsTime).astype(np.bool_)
                        initState = SearchState(newCarPos, eventsPos, eventsTime, eventsCloseTime, eventReward,
                                                canceledEventPenalty,
                                                eventsCanceled, eventsAnswered, float('inf'), 0, None, astarWeight)
                        p = aStar(initState)
                        stochasticCost[numRun] = p[-1].gval
                        if shouldPrint:
                            print('num stochastic run is: ' + str(numRun) + '/' + str(len(stochasticEventsList)))
                            print('a star results is:' + str(stochasticCost[numRun]))
                    optionalExpectedCost[j] = np.mean(stochasticCost)
                    if shouldPrint:
                        print('num Checked move is: ' + str(j) + '/' + str(moveMat.shape[0]))
                        print('expected cost is: ' + str(optionalExpectedCost[j]))
                optionalTotalCost = optionalActionCost + optionalExpectedCost + totalCost
                chosenCost      = np.min(optionalTotalCost)
                chosenMoveIndex = np.argmin(optionalTotalCost)
                chosenRealCost = optionalActionCost[chosenMoveIndex]
                chosenCarMoves = moveMat[chosenMoveIndex, :, :]
                cars.poistions[relevantCarsIndex] = cars.poistions[relevantCarsIndex] + chosenCarMoves
            cars.path.append(cars.poistions)
            totalCost += chosenRealCost
        events.answered,events.timeAnswered = updateEventsAnsweredStatus(cars, events,continuesTimeLine[timeIndex])
        print('finished time:'+str(continuesTimeLine[timeIndex]))
        timeIndex += 1
        # dump logs
        with open('logAnticipatory_' + str(eventReward) + 'EventReward_' + str(gridHeight) + 'grid_' + str(
                numCars) + 'cars_' + str(lengthSim) + 'simLengh_' + str(
                numStochasticRuns) + 'StochasticLength_' + str(lengthPrediction) + 'Prediction_' + str(
                astarWeight) + 'aStarWeight.p', 'wb') as out:
            pickle.dump({'cars': cars,
                         'events': events,
                         'gridSize': gridHeight,
                         'simLength': lengthSim,
                         'cost': totalCost,
                         'numStochasticRuns': numStochasticRuns,
                         'lengthPrediction': lengthPrediction}, out)

    return





if __name__=='__main__':
    main()
    print('Done.')

