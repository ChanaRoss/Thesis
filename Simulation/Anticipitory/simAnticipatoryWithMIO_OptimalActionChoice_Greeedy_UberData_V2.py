# for stats on running time
import time,sys,pickle
from enum import Enum
from copy import deepcopy
import os
# for mathematical calculations and statistical distributions
from scipy.optimize import linear_sum_assignment
from scipy.stats import truncnorm
from scipy.spatial.distance import cdist
from scipy.special import comb
import math
import itertools
import numpy as np
# for graphics
import seaborn as sns
from matplotlib import pyplot as plt
import imageio
sns.set()
# my files
sys.path.insert(0, '/Users/chanaross/dev/Thesis/Simulation/Anticipitory/')
from calculateOptimalActions import runActionOpt, getActions
sys.path.insert(0, '/Users/chanaross/dev/Thesis/MixedIntegerOptimization/')
from offlineOptimizationProblem_TimeWindow import runMaxFlowOpt,plotResults
sys.path.insert(0, '/Users/chanaross/dev/Thesis/UtilsCode/')
from createGif import create_gif

# from simAnticipatoryWithMIO_V1 import Status

class Status(Enum):
    PREOPENED           = 0
    OPENED_COMMITED     = 1
    OPENED_NOT_COMMITED = 2
    CLOSED              = 3
    CANCELED            = 4


class Car:
    def __init__(self, position, id):
        """
        :param position: cartesian coordinate tuple
        :param id: integeral id, uniqe with respect to other cars in simulation
        """
        self.position   = np.reshape(position, (2,))
        self.id         = id
        self.commited   = False
        self.targetId   = None
        self.path       = [deepcopy(self.position)]
        return

    def __lt__(self, other):
        if not isinstance(other, Car):
            raise Exception("other must be of type Car.")
        else:
            return self.id <= other.id

    def __eq__(self, other):
        if not isinstance(other, Car):
            raise Exception("other must be of type Car.")
        elif self.id == other.id:
            raise Exception("Duplicate car Ids used.")
        else:
            return False

    def __hash__(self):
        return hash((self.id, self.position[0], self.position[1]))

    def createCopy(self):
        " return a deep copy of the car"
        return deepcopy(self)

    def commit(self, targetId):
        """
        commits a car to the target
        :param targetId: event to commit car to
        :return: None
        """
        self.commited = True
        self.targetId = targetId
        return

    def uncommit(self):
        self.commited = False
        self.targetId = None

    def fullEq(self, other):
        psEq =  np.array_equal(self.position, other.position)
        idEq = self.id == other.id
        cmEq = self.commited == other.commited
        trEq = self.targetId == other.targetId
        return psEq and idEq and cmEq and trEq


class Event:
    def __init__(self, position, id, start, end):
        """
        :param position: xcartesian coordinate tuple
        :param id: integral id, unique with respect to other events
        :param start: integral start time
        :param end: integral end time
        """
        self.id         = id
        self.position   = np.reshape(position, (2,))
        self.commited   = False
        self.startTime  = start
        self.endTime    = end
        self.status     = Status.PREOPENED
        return

    def __lt__(self, other):
        if not isinstance(other, Event):
            raise Exception("Other must be of type Event")
        else:
            return self.id<=other.id

    def __eq__(self, other):
        if not isinstance(other, Event):
            raise Exception("other must be of type Event.")
        elif self.id==other.id:
            raise Exception("Duplicate event Ids used.")
        else:
            return False

    def __hash__(self):
        return hash(self.id)

    def commit(self, timeDelta):
        self.commited = True
        self.endTime += timeDelta
        return

    def createCopy(self):
        return deepcopy(self)

    def updateStatus(self,currentTime):
        prev = self.status
        if self.status == Status.CLOSED or self.status == Status.CANCELED:
            pass
        elif self.startTime<=currentTime and self.endTime>=currentTime and self.commited:
            self.status = Status.OPENED_COMMITED
        elif self.startTime<=currentTime and self.endTime>=currentTime and not self.commited:
            self.status = Status.OPENED_NOT_COMMITED
        elif self.endTime<currentTime:
            self.status = Status.CANCELED
        else:
            pass
        if self.status != prev:
            return self.status
        else:
            return None

    def fullEq(self, other):
        idEq = self.id == other.id
        psEq = np.array_equal(self.position, other.position)
        cmEq = self.commited == other.commited
        stEq = self.startTime == other.startTime
        etEq = self.endTime == other.endTime
        sttEq = self.status == other.status
        return idEq and psEq and cmEq and stEq and etEq and sttEq


class commitMonitor:
    def __init__(self,commitedDict,notCommitedDict):
        self.commited = commitedDict
        self.notCommited = notCommitedDict

    def commitObject(self, id):
        if id in self.commited.keys():
            raise Exception("error, object is already in commited Dict!")
        else:
            objectToCommit = self.notCommited.pop(id)
            self.commited[id] = objectToCommit

    def unCommitObject(self, id):
        if id in self.notCommited.keys():
            raise Exception("error, object is already in not commited Dict!")
        else:
            objectToUnCommit = self.commited.pop(id)
            self.notCommited[id] = objectToUnCommit

    def getObject(self, id):
        if id in self.commited:
            return self.commited[id]
        elif id in self.notCommited:
            return self.notCommited[id]
        else:
            raise Exception("no object with given id.")

    def removeObject(self,id):
        """
        this function removes id of spesific object, this is used in order to create events with only opened events
        :param id: id wanted to remove
        :return: commit manager without the id which is removed
        """
        if id in self.commited:
            del self.commited[id]
        elif id in self.notCommited:
            del self.notCommited[id]
        else:
            raise Exception("no object with given id.")


    def getCommitedKeys(self):
        return self.commited.keys()

    def getUnCommitedKeys(self):
        return self.notCommited.keys()

    def length(self):
        return len(self.commited)+len(self.notCommited)


class State:
    def __init__(self,root , carMonitor, eventMonitor, cost, parent, time, openedCommitedPenalty = 1,openedNotCommitedPenalty=5, cancelPenalty=50, closeReward=10, timeDelta=5, eps=0.001):
        self.cars                     = carMonitor
        self.events                   = eventMonitor
        self.gval                     = cost
        self.cancelPenalty            = cancelPenalty
        self.closeReward              = closeReward
        self.openedCommitedPenalty    = openedCommitedPenalty
        self.openedNotCommitedPenalty = openedNotCommitedPenalty
        self.parent                   = parent
        self.root                     = root
        self.td                       = timeDelta
        self.time                     = time
        self.eps                      = eps
        self.optionalGval             = 0
        return

    def __lt__(self, other):
        if not isinstance(other, State):
            raise Exception("States must be compared to another State object.")
        else:
            return self.getGval()<=other.getGval()

    def __eq__(self, other):
        if not isinstance(other, State):
            raise Exception("States must be compared to another State object.")
        else:
            # check time equality
            if self.time!=other.time:
                return False
            # check commited car equality
            for k in self.cars.getCommitedKeys():
                if self.cars.getObject(k).fullEq(other.cars.getObject(k)):
                    continue
                else:
                    return False
            # check not commited car equality
            for k in self.cars.getUnCommitedKeys():
                if self.cars.getObject(k).fullEq(other.cars.getObject(k)):
                    continue
                else:
                    return False
            # check commited events equality
            for k in self.events.getCommitedKeys():
                if self.events.getObject(k).fullEq(other.events.getObject(k)):
                    continue
                else:
                    return False
            # check not commited events equality
            for k in self.events.getUnCommitedKeys():
                if self.events.getObject(k).fullEq(other.events.getObject(k)):
                    continue
                else:
                    return False
            # all checks passed
            return True

    def __hash__(self):
        cmHash = [hash(self.cars.getObject(k)) for k in self.cars.getCommitedKeys()]
        ncmHash = [hash(self.cars.getObject(k)) for k in self.cars.getUnCommitedKeys()]
        return hash(tuple(cmHash+ncmHash))

    def getGval(self):
        return self.gval

    def path(self):
        current = self
        p = []
        while current is not None:
            if not current.root:
                p.append(current)
                current = current.parent
            else:
                p.append(current)
                p.reverse()
                return p

    def goalCheck(self):
        # check commited events
        for k in self.events.getCommitedKeys():
            opened  = self.events.getObject(k).status == Status.OPENED_COMMITED
            pre     = self.events.getObject(k).status == Status.PREOPENED
            if opened or pre:
                return False
        # check uncommited events
        for k in self.events.getUnCommitedKeys():
            opened  = self.events.getObject(k).status == Status.OPENED_NOT_COMMITED
            pre     = self.events.getObject(k).status == Status.PREOPENED
            if opened or pre:
                return False
        # all are either closed or canceled
        return True

    def commitCars(self, commit):
        for carId,eventId in commit:
            # update car
            self.cars.getObject(carId).commit(eventId)
            self.cars.commitObject(carId)
            # update event
            self.events.getObject(eventId).commit(self.td)
            self.events.commitObject(eventId)
        return

    def updateOptionalCost(self,commit,matrix):
        # calculate cost of commited events assuming the car reaches the commited event in the following steps
        # commited events
        h1 = 0
        for carId,eventId in commit:
            closeTime = self.events.getObject(eventId).endTime - self.time
            dist = matrix[carId, eventId]
            if dist <= closeTime:
                h1 -= self.closeReward
            else:
                h1 += self.cancelPenalty
        self.optionalGval = h1
        return


    def moveCars(self, move):
        # uncommited
        for i,k in enumerate(self.cars.getUnCommitedKeys()):
            tempCar  = self.cars.getObject(k)
            tempCar.path.append(deepcopy(tempCar.position))
            tempCar.position += move[i, :]
        # commited cars
        for k in self.cars.getCommitedKeys():
            tempCar = self.cars.getObject(k)
            tempCar.path.append(deepcopy(tempCar.position))
            targetPosition = self.events.getObject(tempCar.targetId).position
            delta = targetPosition - tempCar.position
            if delta[0]!= 0:
                tempCar.position[0] += np.sign(delta[0])
            else:
                tempCar.position[1] += np.sign(delta[1])
        return

    def updateStatus(self, matrix):
        # update event status
        counter = {Status.OPENED_COMMITED: 0, Status.OPENED_NOT_COMMITED: 0, Status.CLOSED: 0, Status.CANCELED: 0}
        for eventId in range(self.events.length()):
            tempEvent               = self.events.getObject(eventId)
            newStatus               = tempEvent.updateStatus(self.time)
            if newStatus is not None:
                counter[newStatus] += 1

        # update commited cars and events
        for carId in range(self.cars.length()):
            tempCar = self.cars.getObject(carId)
            if tempCar.commited:
                if matrix[carId, tempCar.targetId] <= self.eps and self.events.getObject(tempCar.targetId).status == Status.OPENED_COMMITED:
                    self.events.getObject(tempCar.targetId).status = Status.CLOSED  # update event status
                    self.events.unCommitObject(tempCar.targetId)  # uncommit event
                    self.cars.unCommitObject(carId)  # uncommit car
                    tempCar.uncommit()  # update car field
                    counter[Status.CLOSED] += 1
            else:  # update uncommited events
                closedEvents = np.where(matrix[carId, :] <= self.eps)
                for e in closedEvents[0]:
                    tempEvent = self.events.getObject(e)
                    if tempEvent.status == Status.OPENED_NOT_COMMITED:  # uncommited event reached
                        tempEvent.status = Status.CLOSED  # close event
                        counter[Status.CLOSED] += 1  # increment counter
                        break
        return counter


    def getDistanceMatrix(self):
        # create distance matrix
        carPositions = np.vstack([self.cars.getObject(i).position for i in range(self.cars.length())])
        evePositions = np.vstack([self.events.getObject(i).position for i in range(self.events.length())])
        matrix = cdist(carPositions, evePositions, metric="cityblock")
        return matrix

    def updateCost(self, counter, move):
        cost = 0
        cost += np.sum(abs(move))
        cost += len(self.cars.commited)
        cost += counter[Status.OPENED_COMMITED]*self.openedCommitedPenalty
        cost += counter[Status.OPENED_NOT_COMMITED]*self.openedNotCommitedPenalty
        cost += counter[Status.CANCELED]*self.cancelPenalty
        cost -= counter[Status.CLOSED]*self.closeReward
        self.gval += cost
        return


def moveGenerator(numCars):
    moveIter = itertools.product([(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)], repeat=numCars)
    for move in moveIter:
        yield np.array(move).astype(np.int8)


def moveGeneratorIndexs(numCars, actionOptions):
    moveIter = itertools.product(range(len(actionOptions)), repeat=numCars)
    for move in moveIter:
        yield np.array(move).astype(np.int8)


def commitGenerator(carIdList, eventIdList):
    if len(eventIdList):
        numOptionalCommits = np.min([len(carIdList)+1, len(eventIdList)+1])
        for i in range(numOptionalCommits):
            for carChoice in itertools.combinations(carIdList, i):
                for eventChoice in itertools.combinations(eventIdList, i):
                    for eventChoicePermutations in itertools.permutations(eventChoice, len(eventChoice)):
                        yield list(zip(carChoice, eventChoicePermutations))
    else:
        yield([])


def descendantGenerator(state):
    tempEventList = list(state.events.getUnCommitedKeys())
    eventList = [e for e in tempEventList if state.events.getObject(e).status == Status.OPENED_NOT_COMMITED]
    carList   = list(state.cars.getUnCommitedKeys())
    commitGen = commitGenerator(carList, [])
    for commit in commitGen:
        # commit cars
        newCommitState = deepcopy(state)
        # update root,parent and time for all descendants
        newCommitState.root     = False
        newCommitState.parent   = state
        newCommitState.time     += 1
        newCommitState.commitCars(commit)
        newCommitState.updateOptionalCost(commit,newCommitState.getDistanceMatrix())
        numUnCommitedCars = len(newCommitState.cars.notCommited)
        moveGen = moveGenerator(numUnCommitedCars)
        for possibleMove in moveGen:
            # create copy for new descendant
            newMoveState = deepcopy(newCommitState)
            # move the cars according to possible move
            newMoveState.moveCars(possibleMove)
            # calc distance matrix between cars and events
            dm = newMoveState.getDistanceMatrix()
            # update status of events relative to new car positions
            counter = newMoveState.updateStatus(dm)
            # update cost: sum of new cost and previous state cost
            newMoveState.updateCost(counter, possibleMove)
            # yield the new state
            yield (newMoveState,possibleMove)


def singleCarMovementGenerator(state, actionDict, carIndex):
    nCars   = state.cars.length()
    for i in range(5):
        possibleMove = np.zeros(shape=(nCars, 2)).astype(int)
        possibleMove[carIndex, :] = actionDict[i]
        # create copy for new descendant
        newMoveState = deepcopy(state)
        newMoveState.root = False
        newMoveState.parent = state
        newMoveState.time += 1
        # move the cars according to possible move
        newMoveState.moveCars(possibleMove)
        # calc distance matrix between cars and events
        dm = newMoveState.getDistanceMatrix()
        # update status of events relative to new car positions
        counter = newMoveState.updateStatus(dm)
        # update cost: sum of new cost and previous state cost
        newMoveState.updateCost(counter, possibleMove)
        # yield the new state
        yield newMoveState


def spesificDescendantGenerator(state, possibleMoves):
    """
    this function calculates the cost of preforming a certain action assuming you move the cars each in their direction
    :param state: the previous state before the cars were moved
    :param possibleMoves: np array 2*n_c (movement of each car in each direction)
    :return: cost of preforming the movement
    """
    # create copy for new descendant
    tempState        = deepcopy(state)
    tempState.root   = False
    tempState.parent = state
    tempState.time  += 1
    # move the cars according to possible move
    tempState.moveCars(possibleMoves)
    # calc distance matrix between cars and events
    dm               = tempState.getDistanceMatrix()
    # update status of events relative to new car positions
    counter          = tempState.updateStatus(dm)
    # update cost: sum of new cost and previous state cost
    tempState.updateCost(counter,possibleMoves)
    return tempState




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


def createEventDistributionUber(simStartTime, startTime, endTime, probabilityMatrix,eventTimeWindow,simTime):
    eventPos = []
    eventTimes = []
    firstTime = startTime + simTime + simStartTime  # each time is a 5 min
    numTimeSteps = endTime - startTime
    for t in range(numTimeSteps):
        for x in range(probabilityMatrix.shape[0]):
            for y in range(probabilityMatrix.shape[1]):
                randNum = np.random.uniform(0, 1)
                cdfNumEvents = probabilityMatrix[x, y, t + firstTime, :]
                # find how many events are happening at the same time
                numEvents = np.searchsorted(cdfNumEvents, randNum, side='left')
                numEvents = np.floor(numEvents).astype(int)
                # print('at loc:' + str(x) + ',' + str(y) + ' num events:' + str(numEvents))
                #for n in range(numEvents):
                if numEvents > 0:
                    eventPos.append(np.array([x, y]))
                    eventTimes.append(t + startTime)
    eventsPos  = np.array(eventPos)
    eventTimes = np.array(eventTimes)
    eventsTimeWindow = np.column_stack([eventTimes, eventTimes + eventTimeWindow])
    print('number of events created:'+str(eventsPos.shape[0]))
    return eventsPos, eventsTimeWindow

def createRealEventsDistributionUber(simStartTime, startTime, endTime, eventsMatrix,eventTimeWindow,simTime):
    eventPos = []
    eventTimes = []
    firstTime = startTime + simTime + simStartTime  # each time is a 5 min
    numTimeSteps = endTime - startTime
    for t in range(numTimeSteps):
        for x in range(eventsMatrix.shape[0]):
            for y in range(eventsMatrix.shape[1]):
                randNum = np.random.uniform(0, 1)
                numEvents = eventsMatrix[x, y, t + firstTime]
                # print('at loc:' + str(x) + ',' + str(y) + ' num events:' + str(numEvents))
                # for n in range(numEvents):
                if numEvents > 0:
                    eventPos.append(np.array([x, y]))
                    eventTimes.append(t + startTime)
    eventsPos = np.array(eventPos)
    eventTimes = np.array(eventTimes)
    eventsTimeWindow = np.column_stack([eventTimes, eventTimes + eventTimeWindow])
    print('number of events created:' + str(eventsPos.shape[0]))
    return eventsPos, eventsTimeWindow

def createEventsDistribution(gridSize, startTime, endTime, lam, eventTimeWindow):
    locX        = gridSize / 2
    scaleX      = gridSize / 3
    locY        = gridSize / 2
    scaleY      = gridSize / 3
    # randomize event times
    eventTimes  = poissonRandomEvents(startTime, endTime, lam)
    eventPosX   = truncnorm.rvs((0 - locX) / scaleX, (gridSize - locX) / scaleX, loc=locX, scale=scaleX,
                              size=len(eventTimes)).astype(np.int64)
    eventPosY   = truncnorm.rvs((0 - locY) / scaleY, (gridSize - locY) / scaleY, loc=locY, scale=scaleY,
                              size=len(eventTimes)).astype(np.int64)

    eventsPos           = np.column_stack([eventPosX,eventPosY])
    eventsTimeWindow    = np.column_stack([eventTimes,eventTimes+eventTimeWindow])
    return eventsPos,eventsTimeWindow


def createStochasticEvents(simStartTime, numStochasticRuns, startTime, endTime, probabilityMatrix, eventsTimeWindow, simTime):
    """
    this function creates stochastic events with time and position and adds them to dictionary
    :param numStochasticRuns: number of stochastic runs to create
    :param startingId: the id from which to start adding events (num opened not commited events)
    :param startTime: time from which to start stochastic events
    :param endTime: last time for events
    :param lam: rate of events for poisson distribution
    :param eventsTimeWindow: time each event is opened
    :return: Dictionary of stochastic runs, in each stochastic run is the dictionary of events
    """
    stochasticEventsDict = {}
    for i in range(numStochasticRuns):
        eventPos, eventTimeWindow = createEventDistributionUber(simStartTime, startTime, endTime, probabilityMatrix, eventsTimeWindow, simTime)
        stochasticEventsDict[i] = {'eventsPos': eventPos, 'eventsTimeWindow': eventTimeWindow}
    return stochasticEventsDict


def anticipatorySimulation(initState, nStochastic, gs, tPred, eTimeWindow, simStartTime, probabilityMatrix, shouldPrint=False):
    """
    this function is the anticipatory simulation
    :param initState: initial state of the system (cars and events)
    :param nStochastic: number of stochastic runs
    :param gs: grid size (X*Y)
    :param tPred: number of time steps for prediction
    :param eTimeWindow: number of time steps an event is opened
    :param lam: rate of events for poission distribution
    :param shouldPrint: True/False if the code should print log
    :return:
    """
    isGoal  = False
    current = initState
    optionalActions        = np.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]).astype(int)
    optionalActionsToIndex = {(0,  0): 0,
                              (0,  1): 1,
                              (0, -1): 2,
                              (1,  0): 3,
                              (-1, 0): 4}
    while not isGoal:
        currentTime = current.time
        stochasticEventsDict = createStochasticEvents(simStartTime, nStochastic, 1, 1 + tPred, probabilityMatrix, eTimeWindow, currentTime)
        nCars                = current.cars.length()
        moveOptions          = np.array(list(moveGeneratorIndexs(2, optionalActions)))
        possibleMoves        = np.zeros(shape=(nCars, 2))
        coupledRealCost      = np.zeros(shape=(nCars, nCars, 25))
        # calculate the cost of moving two cars out of the total cars.
        for carsIndex in itertools.permutations(range(nCars), 2):
            for i, moveOption in enumerate(moveOptions):  # movement options for two cars
                carMoves = deepcopy(possibleMoves)
                carMoves[carsIndex[0], :] = optionalActions[moveOption[0]]
                carMoves[carsIndex[1], :] = optionalActions[moveOption[1]]
                tempState = spesificDescendantGenerator(current, carMoves.astype(int))
                coupledRealCost[carsIndex[0], carsIndex[1], i] = tempState.gval

        # calculate the expected value of moving a single car
        # size is number of cars and number of optional actions for each car
        expectedCost        = np.zeros(shape=(nCars, len(optionalActions)))
        totalExpectedCost   = np.zeros(shape=(nCars, len(optionalActions)))
        for carIndex in range(nCars):
            # for each car calculate the optional expected cost of moving that car in some direction
            for actionIndex, tempOptionalState in enumerate(singleCarMovementGenerator(current, optionalActions, carIndex)):
                stochasticCost = np.zeros(shape=(nStochastic, 1))
                # initialize variables for stochastic optimization
                carsPos                 = np.zeros(shape=(tempOptionalState.cars.length(), 2))
                currentEventsPos        = []
                currentEventStartTime   = []
                currentEventsEndTime    = []
                # get car locations from state -
                for d, k in enumerate(tempOptionalState.cars.getUnCommitedKeys()):
                    carsPos[d, :] = deepcopy(tempOptionalState.cars.getObject(k).position)
                # get opened event locations from state -
                for k in tempOptionalState.events.getUnCommitedKeys():
                    if tempOptionalState.events.getObject(k).status == Status.OPENED_NOT_COMMITED:
                        currentEventsPos.append(deepcopy(tempOptionalState.events.getObject(k).position))
                        # assume that event start time is current time for deterministic runs and the time left for event
                        # is the time left - current time.
                        # the deterministic run is from (currentTime+1) therefore need to subtract that value and not CurrentTime
                        currentEventStartTime.append(deepcopy(tempOptionalState.events.getObject(k).startTime) - (currentTime + 1))
                        currentEventsEndTime.append(deepcopy(tempOptionalState.events.getObject(k).endTime) - (currentTime + 1))
                # run deterministic optimization for stochastic events -
                for j in range(len(stochasticEventsDict)):
                    if len(stochasticEventsDict[j]['eventsPos']) + len(currentEventsPos) > 0:
                        # there are events to be tested in deterministic optimization:
                        eventsPos           = deepcopy(currentEventsPos)
                        eventsStartTime     = deepcopy(currentEventStartTime)
                        eventsEndTime       = deepcopy(currentEventsEndTime)
                        temp = [eventsPos.append(e) for e in stochasticEventsDict[j]['eventsPos']]
                        temp = [eventsStartTime.append(e[0]) for e in stochasticEventsDict[j]['eventsTimeWindow']]
                        temp = [eventsEndTime.append(e[1]) for e in stochasticEventsDict[j]['eventsTimeWindow']]
                        eventsPos           = np.array(eventsPos).reshape(len(eventsPos), 2)
                        eventsStartTime     = np.array(eventsStartTime)
                        eventsEndTime       = np.array(eventsEndTime)
                        stime               = time.process_time()
                        m, obj              = runMaxFlowOpt(0, carsPos, eventsPos, eventsStartTime, eventsEndTime,
                                            tempOptionalState.closeReward, tempOptionalState.cancelPenalty,
                                            tempOptionalState.openedNotCommitedPenalty, 0)
                        etime   = time.process_time()
                        runTime = etime - stime
                        try:
                            stochasticCost[j] = -obj.getValue()
                        except:
                            print('failed cuz of gurobi!')
                    else:
                        stochasticCost[j] = 0
                # calculate expected cost of all stochastic runs for this spesific optional State
                if shouldPrint:
                    print("stochastic cost of optional run is:")
                    print(np.transpose(stochasticCost))
                expectedCost[carIndex, actionIndex] = np.mean(stochasticCost)
                # optional total cost includes the actual cost of movement + optional cost for commited events + expected cost of future events
                totalExpectedCost[carIndex, actionIndex] = expectedCost[carIndex, actionIndex] + tempOptionalState.gval + tempOptionalState.optionalGval
                if shouldPrint:
                    print('state cost is: ' + str(tempOptionalState.gval) + ', expected cost is: ' + str(expectedCost[carIndex, actionIndex]) + ' , commited cost is:' + str(tempOptionalState.optionalGval))

        m2, obj2 = runActionOpt(coupledRealCost, totalExpectedCost, outputFlag=0)
        chosenActionsIndex = getActions(m2, nCars)
        for i in range(nCars):
            possibleMoves[i, :] = optionalActions[np.where(chosenActionsIndex[i, :] == 1)]

        chosenState = spesificDescendantGenerator(current, possibleMoves.astype(int))
        current     = chosenState
        print('t:' + str(currentTime) + ' , chosen cost is: ' + str(obj2.getValue()))

        # check if this state is a goal or not-
        if current.goalCheck():
            isGoal = True
            print('finished run - total cost is:' + str(current.gval))
        # dump logs
        dataInRun = postAnalysis(current.path())
        # Anticipatory output:
        # with open('SimAnticipatoryMioResults_' + str(currentTime + 1) + 'time_' + str(
        #         current.events.length()) + 'numEvents_' + str(current.cars.length()) + 'numCars_uberData.p',
        #           'wb') as out:
        #     pickle.dump({'pathresults': current.path(),
        #                  'time': dataInRun['timeVector'],
        #                  'gs': gs,
        #                  'OpenedEvents': dataInRun['openedEvents'],
        #                  'closedEvents': dataInRun['closedEvents'],
        #                  'canceledEvents': dataInRun['canceledEvents'],
        #                  'allEvents': dataInRun['allEvents'],
        #                  'stochasticResults': optionalTotalCost,
        #                  'stochasticEventsDict': stochasticEventsDict,
        #                  'cost': current.gval}, out)
    return current.path()


def greedySimulation(initState, shouldPrint):
    isGoal = False
    current = initState
    while not isGoal:
        currentTime             = current.time
        newState                = deepcopy(current)
        # update root,parent and time for all descendants
        newState.root           = False
        newState.parent         = current
        newState.time          += 1
        carsPos                 = np.zeros(shape=(newState.cars.length(), 2))
        currentEventsPos        = []
        currentEventStartTime   = []
        currentEventsEndTime    = []
        currentEventIndex       = []
        carMoves                = []
        # get car locations from state -
        for d, k in enumerate(newState.cars.getUnCommitedKeys()):
            carsPos[d, :] = deepcopy(newState.cars.getObject(k).position)
            newState.cars.getObject(k).path.append(deepcopy(carsPos[d, :]))
        # get opened event locations from state -
        for k in newState.events.getUnCommitedKeys():
            if newState.events.getObject(k).status == Status.OPENED_NOT_COMMITED:
                currentEventsPos.append(deepcopy(newState.events.getObject(k).position))
                currentEventIndex.append(deepcopy(newState.events.getObject(k).id))
                currentEventStartTime.append(deepcopy(newState.events.getObject(k).startTime))
                currentEventsEndTime.append(deepcopy(newState.events.getObject(k).endTime))
        carsPos          = np.array(carsPos)
        currentEventsPos = np.array(currentEventsPos)
        # run deterministic optimization for stochastic events -
        if currentEventsPos.shape[0]>0:
            distMat = cdist(carsPos,currentEventsPos,metric='cityblock')
            # matching
            carIndices, matchedIndices = linear_sum_assignment(distMat)
            for ie,ic in enumerate(carIndices):
                tempCar = newState.cars.getObject(ic)
                targetPosition = newState.events.getObject(currentEventIndex[matchedIndices[ie]]).position
                delta = targetPosition - tempCar.position
                if delta[0] != 0:
                    carMovement = np.array([np.sign(delta[0]),0])

                else:
                    carMovement = np.array([0,np.sign(delta[1])])
                tempCar.position += carMovement
                carMoves.append(carMovement)
                if shouldPrint:
                    print('t:'+str(currentTime+1)+'moved car:'+str(ic)+' towards event:'+str(currentEventIndex[ie]))
        else:
            for i in range(newState.cars.length()):
                # there are no opened events so cars should not move
                carMoves.append(np.array([0,0]))
        # calc distance matrix between cars and events
        dm = newState.getDistanceMatrix()
        # update status of events relative to new car positions
        counter = newState.updateStatus(dm)
        # update cost: sum of new cost and previous state cost
        newState.updateCost(counter, np.array(carMoves))
        current = newState
        # check if this state is a goal or not-
        if current.goalCheck():
            isGoal = True
            print('finished run - total cost is:' + str(current.gval))
        # dump logs
        # with open('SimGreedy' + str(currentTime + 1) + 'time_' + str(
        #         current.events.length()) + 'numEvents_' + str(current.cars.length()) + 'numCars_uberData.p', 'wb') as out:
        #     pickle.dump({'time': currentTime + 1,
        #                  'currentState': current,
        #                  'cost': current.gval}, out)
    return current.path()



def optimizedSimulation(initialState, fileLoc, fileName, gridSize):
    plotFigures     = False
    carsPos         = np.zeros(shape=(initialState.cars.length(), 2))
    eventsPos       = []
    eventsStartTime = []
    eventsEndTime   = []
    for d, k in enumerate(initialState.cars.getUnCommitedKeys()):
        carsPos[d, :] = deepcopy(initialState.cars.getObject(k).position)
    # get opened event locations from state -
    for k in initialState.events.getUnCommitedKeys():
        eventsPos.append(deepcopy(initialState.events.getObject(k).position))
        eventsStartTime.append(deepcopy(initialState.events.getObject(k).startTime))
        eventsEndTime.append(deepcopy(initialState.events.getObject(k).endTime))

    m, obj = runMaxFlowOpt(0, carsPos, np.array(eventsPos), np.array(eventsStartTime), np.array(eventsEndTime),
                           initialState.closeReward,initialState.cancelPenalty, initialState.openedNotCommitedPenalty, 0)
    dataOut = plotResults(m, carsPos, np.array(eventsPos), np.array(eventsStartTime), np.array(eventsEndTime), plotFigures, fileLoc
                ,fileName, gridSize)

    dataOut['cost'] = -obj.getValue()
    return dataOut

def main():
    # loading probability matrix from uber data. matrix is: x,y,h where x,y are the grid size and h is the time (0-24 hours)
    probFileName        = '/Users/chanaross/dev/Thesis/ProbabilityFunction/CreateEvents/4D_UpdatedGrid_5min_250grid_LimitedProbability_CDFMat_wday_1.p'
    probabilityMatrix   = np.load(probFileName)
    xLim                = [0, 20]
    yLim                = [40, 70]
    probabilityMatrix   = probabilityMatrix[xLim[0]:xLim[1], yLim[0]:yLim[1], :, :]
    eventsFileName      = '/Users/chanaross/dev/Thesis/UberData/4D_UpdatedGrid_5min_250Grid_LimitedEventsMat_wday1.p'
    eventsMatrix        = np.load(eventsFileName)
    eventsMatrix        = eventsMatrix[xLim[0]:xLim[1], yLim[0]:yLim[1], :, 1]

    np.random.seed(50)
    shouldRunAnticipatory   = 1
    shouldRunGreedy         = 1
    shouldRunOptimization   = 1
    loadFromPickle          = 0
    shouldPrint             = False
    # params
    epsilon                     = 0.001  # distance between locations to be considered same location
    simStartTime                = 0
    lengthSim                   = 24 * 3  # one hour, each time step is 5 min. of real time
    numStochasticRuns           = 160
    lengthPrediction            = 7
    deltaTimeForCommit          = 10
    closeReward                 = 80
    cancelPenalty               = 140
    openedCommitedPenalty       = 1
    openedNotCommitedPenalty    = 5

    gridSize            = [probabilityMatrix.shape[0], probabilityMatrix.shape[1]]
    deltaOpenTime       = 3
    numCars             = 5
    carPosX             = np.random.randint(0, gridSize[0], numCars)
    carPosY             = np.random.randint(0, gridSize[1], numCars)
    carPos              = np.column_stack((carPosX, carPosY)).reshape(numCars, 2)

    eventPos,eventTimes = createRealEventsDistributionUber(simStartTime, 0, lengthSim, eventsMatrix, deltaOpenTime, 0)
    eventStartTime      = eventTimes[:, 0]
    eventEndTime        = eventTimes[:, 1]


    # plt.scatter(eventPos[:,0],eventPos[:,1], c= 'r')
    # plt.scatter(carPos[:,0],carPos[:,1],c='k')
    # plt.show()
    uncommitedCarDict   = {}
    commitedCarDict     = {}
    uncommitedEventDict = {}
    commitedEventDict   = {}
    numEvents           = eventStartTime.shape[0]
    for i in range(numCars):
        tempCar = Car(carPos[i,:], i)
        uncommitedCarDict[i]   = tempCar
    for i in range(numEvents):
        tempEvent = Event(eventPos[i, :], i, eventStartTime[i], eventEndTime[i])
        uncommitedEventDict[i] = tempEvent

    carMonitor   = commitMonitor(commitedCarDict, uncommitedCarDict)
    eventMonitor = commitMonitor(commitedEventDict, uncommitedEventDict)
    initState    = State(root=True, carMonitor=carMonitor, eventMonitor=eventMonitor, cost=0, parent=None,
                         time=0, openedNotCommitedPenalty = openedNotCommitedPenalty, openedCommitedPenalty = openedCommitedPenalty,
                      cancelPenalty=cancelPenalty, closeReward=closeReward,
                      timeDelta=deltaTimeForCommit,eps=epsilon)
    fileName = str(lengthPrediction) + 'lpred_' + str(simStartTime) + 'startTime_' + str(gridSize[0]) + 'gridX_' +\
               str(gridSize[1]) + 'gridY_' + str(eventTimes.shape[0]) + 'numEvents_' + \
               str(numStochasticRuns) + 'nStochastic_' + str(numCars) + 'numCars_uberData'
    fileLoc = '/Users/chanaross/dev/Thesis/Simulation/Anticipitory/Results/'

    if shouldRunAnticipatory:
        # run anticipatory:
        stime           = time.process_time()
        pAnticipatory   = anticipatorySimulation(initState, numStochasticRuns, gridSize, lengthPrediction, deltaOpenTime, simStartTime, probabilityMatrix, shouldPrint=shouldPrint)
        etime           = time.process_time()
        runTimeA        = etime - stime
        print('Anticipatory cost is:' + str(pAnticipatory[-1].gval))
        print('run time is:'+str(runTimeA))


        dataAnticipatory = postAnalysis(pAnticipatory)
        anticipatoryFileName = 'SimAnticipatory_OptimalActionChoice_MioFinalResults_'+fileName
        greedyFileName       = 'SimGreedyFinalResults_'+fileName
        # Anticipatory output:
        with open('SimAnticipatory_OptimalActionChoice_MioFinalResults_' + fileName+'.p', 'wb') as out:
            pickle.dump({'runTime'          : runTimeA,
                         'pathresults'      : pAnticipatory,
                         'time'             : dataAnticipatory['timeVector'],
                         'gs'               : gridSize,
                         'OpenedEvents'     : dataAnticipatory['openedEvents'],
                         'closedEvents'     : dataAnticipatory['closedEvents'],
                         'canceledEvents'   : dataAnticipatory['canceledEvents'],
                         'allEvents'        : dataAnticipatory['allEvents'],
                         'cost'             : pAnticipatory[-1].gval}, out)
        time2 = []
        if not os.path.isdir(fileLoc + anticipatoryFileName):
            os.mkdir(fileLoc + anticipatoryFileName)
        for s in pAnticipatory:
            plotForGif(s, numEvents, numCars, gridSize, fileLoc + '/' + anticipatoryFileName + '/' + anticipatoryFileName)
            time2.append(s.time)
        listNames = [anticipatoryFileName + '_' + str(t) + '.png' for t in time2]
        create_gif(fileLoc + anticipatoryFileName + '/', listNames, 1, anticipatoryFileName)
        plt.close()
    if shouldRunGreedy:
        if loadFromPickle:
            pickleName = 'SimAnticipatoryMioFinalResults_8lpred_0startTime_10gridX_10gridY_23numEvents_30nStochastic_2numCars_uberData'
            pathName = '/home/chana/Documents/Thesis/FromGitFiles/Simulation/Anticipitory/PickleFiles/'
            dataPickle = pickle.load(open(pathName + pickleName + '.p', 'rb'))
            initState = dataPickle['pathresults'][0]
            gridSize  = dataPickle['gs']
            fileName  = '15numEvents_2numCars_0.75lam_7gridSize'
            numEvents = initState.events.length()
            numCars   = initState.cars.length()

        # run greedy:
        stime = time.process_time()
        pGreedy = greedySimulation(initState, True)
        etime = time.process_time()
        runTimeG = etime - stime
        print('Greedy cost is:' + str(pGreedy[-1].gval))
        print('run time is: ' + str(runTimeG))
        dataGreedy = postAnalysis(pGreedy)
        # Greedy output:
        with open('SimGreedyFinalResults_' + fileName + '.p', 'wb') as out:
            pickle.dump({'runTime'          : runTimeG,
                         'pathresults'      : pGreedy,
                         'time'             : dataGreedy['timeVector'],
                         'gs'               : gridSize,
                         'OpenedEvents'     : dataGreedy['openedEvents'],
                         'closedEvents'     : dataGreedy['closedEvents'],
                         'canceledEvents'   : dataGreedy['canceledEvents'],
                         'allEvents'        : dataGreedy['allEvents'],
                         'cost'             : pGreedy[-1].gval}, out)

        time2 = []
        if not os.path.isdir(fileLoc + greedyFileName):
            os.mkdir(fileLoc + greedyFileName)
        for s in pGreedy:
            plotForGif(s, numEvents, numCars, gridSize,
                       fileLoc + '/' + greedyFileName + '/' + greedyFileName)
            time2.append(s.time)
        listNames = [greedyFileName + '_' + str(t) + '.png' for t in time2]
        create_gif(fileLoc + greedyFileName + '/', listNames, 1, greedyFileName)
        plt.close()


    if shouldRunOptimization:
        if loadFromPickle:
            pickleName = 'SimAnticipatory_randomChoice_MioFinalResults_7lpred_0startTime_20gridX_30gridY_63numEvents_100nStochastic_4numCars_uberData'
            pathName = '/Users/chanaross/dev/Thesis/Simulation/Anticipitory/PickleFiles/'
            dataPickle = pickle.load(open(pathName + pickleName + '.p', 'rb'))
            initState = dataPickle['pathresults'][0]
            gridSize  = dataPickle['gs']
            fileName  = '7lpred_0startTime_20gridX_30gridY_63numEvents_100nStochastic_4numCars_uberData'
            numEvents = initState.events.length()
            numCars   = initState.cars.length()

        dataOptimization = optimizedSimulation(initState, fileLoc, fileName, gridSize)

        # optimization output:
        with open('SimOptimizationFinalResults_' + fileName + '.p', 'wb') as out:
            pickle.dump({'time'             : dataOptimization['time'],
                         'gs'               : gridSize,
                         'OpenedEvents'     : dataOptimization['openedEvents'],
                         'closedEvents'     : dataOptimization['closedEvents'],
                         'canceledEvents'   : dataOptimization['canceledEvents'],
                         'cost'             : dataOptimization['cost'],
                         'allEvents'        : dataOptimization['allEvents']}, out)

    return



def postAnalysis(p):
    actualMaxTime   = len(p)
    numEvents       = p[0].events.length()
    numUnCommited   = np.zeros(actualMaxTime)
    numCommited     = np.zeros(actualMaxTime)
    timeVector      = list(range(actualMaxTime))
    closedEvents    = np.zeros(actualMaxTime)
    openedEvents    = np.zeros(actualMaxTime)
    canceledEvents  = np.zeros(actualMaxTime)
    allEvents       = np.zeros(actualMaxTime)
    dataOut         = {}
    for i, st in enumerate(p):
        numUnCommited[i]    = len(list(st.events.getUnCommitedKeys()))
        numCommited[i]      = len(list(st.events.getCommitedKeys()))
        for iE in range(numEvents):
            eventTemp       = st.events.getObject(iE)
            if eventTemp.status == Status.CANCELED:
                canceledEvents[i]   += 1
            elif eventTemp.status == Status.CLOSED:
                closedEvents[i]     += 1
            elif eventTemp.status == Status.OPENED_COMMITED or eventTemp.status == Status.OPENED_NOT_COMMITED:
                openedEvents[i]     += 1
        allEvents[i]        = canceledEvents[i] + closedEvents[i] + openedEvents[i]

    dataOut['closedEvents']     = closedEvents
    dataOut['openedEvents']     = openedEvents
    dataOut['canceledEvents']   = canceledEvents
    dataOut['actualMaxTime']    = actualMaxTime
    dataOut['allEvents']        = allEvents
    dataOut['timeVector']       = timeVector
    return dataOut



def plotForGif(s, ne,nc, gs, fileName):
    """
        plot cars as red points, events as blue points,
        and lines connecting cars to their targets
        :param carDict:
        :param eventDict:
        :return: image for gif
        """
    fig, ax = plt.subplots()
    ax.set_title('time: {0}'.format(s.time))
    for c in range(nc):
        carTemp = s.cars.getObject(c)
        ax.scatter(carTemp.position[0], carTemp.position[1], c='k', alpha=0.5)
    ax.scatter([], [], c='b', marker='*', label='Opened not commited')
    ax.scatter([], [], c='b', label='Opened commited')
    ax.scatter([], [], c='r', label='Canceled')
    ax.scatter([], [], c='g', label='Closed')
    for i in range(ne):
        eventTemp = s.events.getObject(i)
        if eventTemp.status == Status.OPENED_COMMITED:
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='b', alpha=0.7)
        elif eventTemp.status == Status.OPENED_NOT_COMMITED:
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='b', marker='*', alpha=0.7)
        elif (eventTemp.status == Status.CLOSED):
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='g', alpha=0.2)
        elif (eventTemp.status == Status.CANCELED):
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='r', alpha=0.2)
        else:
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='y', alpha=0.2)
    ax.set_xlim([-1, gs[0] + 1])
    ax.set_ylim([-1, gs[1] + 1])
    ax.grid(True)
    plt.legend()
    plt.savefig(fileName + '_' + str(s.time) + '.png')
    plt.close()
    return

if __name__ == '__main__':
    main()
    print('Done.')
