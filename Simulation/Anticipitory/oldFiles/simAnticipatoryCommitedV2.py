# for stats on running time
import time,sys,pickle
from enum import Enum
from copy import deepcopy
# for mathematical calculations and statistical distributions
from scipy.stats import truncnorm
from scipy.spatial.distance import cdist
from scipy.special import comb
import math
import itertools
import numpy as np
# my files
sys.path.insert(0, '/home/chanaby/Documents/Thesis/Thesis/SearchAlgorithm/')
from iterativeAstarV2 import aStar,Status,SearchState
from iterativeAstarV2 import Car as SearchCar
from iterativeAstarV2 import Event as SearchEvent
# for graphics
import seaborn as sns
from matplotlib import pyplot as plt
import imageio

sns.set()




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
        self.path       = [self.position]
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
        elif self.startTime<=currentTime and self.endTime>=currentTime:
            self.status = Status.OPENED
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
    def __init__(self,root , carMonitor, eventMonitor, cost, parent, time, cancelPenalty=50, closeReward=10, timeDelta=5, eps=0.001):
        self.cars           = carMonitor
        self.events         = eventMonitor
        self.gval           = cost
        self.cancelPenalty  = cancelPenalty
        self.closeReward    = closeReward
        self.parent         = parent
        self.root           = root
        self.td             = timeDelta
        self.time           = time
        self.eps            = eps
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
            opened = self.events.getObject(k).status == Status.OPENED
            pre = self.events.getObject(k).status == Status.PREOPENED
            if opened or pre:
                return False
        # check uncommited events
        for k in self.events.getUnCommitedKeys():
            opened = self.events.getObject(k).status == Status.OPENED
            pre = self.events.getObject(k).status == Status.PREOPENED
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

    def moveCars(self, move):
        # uncommited
        for i,k in enumerate(self.cars.getUnCommitedKeys()):
            tempCar  = self.cars.getObject(k)
            tempCar.path.append(tempCar.position)
            tempCar.position += move[i, :]
        # commited cars
        for k in self.cars.getCommitedKeys():
            tempCar = self.cars.getObject(k)
            tempCar.path.append(tempCar.position)
            targetPosition = self.events.getObject(tempCar.targetId).position
            delta = tempCar.position-targetPosition
            if delta[0]!=0:
                tempCar.position[0] += np.sign(delta[0])
            else:
                tempCar.position[1] += np.sign(delta[1])
        return

    def updateStatus(self, matrix):
        # update event status
        counter = {Status.OPENED : 0, Status.CLOSED: 0, Status.CANCELED: 0}
        for eventId in range(self.events.length()):
            tempEvent               = self.events.getObject(eventId)
            newStatus               = tempEvent.updateStatus(self.time)
            if newStatus is not None:
                counter[newStatus] += 1

        # update commited cars and events
        for carId in range(self.cars.length()):
            tempCar = self.cars.getObject(carId)
            if tempCar.commited:
                if matrix[carId, tempCar.targetId]<=self.eps and self.events.getObject(tempCar.targetId).status==Status.OPENED:
                    self.events.getObject(tempCar.targetId).status = Status.CLOSED  # update event status
                    self.events.unCommitObject(tempCar.targetId)  # uncommit event
                    self.cars.unCommitObject(carId)  # uncommit car
                    tempCar.uncommit()  # update car field
                    counter[Status.CLOSED] += 1
            else:  # update uncommited events
                closedEvents = np.where(matrix[carId, :]<=self.eps)
                for e in closedEvents[0]:
                    tempEvent = self.events.getObject(e)
                    if tempEvent.status==Status.OPENED and not tempEvent.commited:  # uncommited event reached
                        tempEvent.status = Status.CLOSED  # close event
                        counter[Status.CLOSED] += 1  # incriment counter
        return counter

    def getStateForStochasticRuns(self,stochasticEventsDict):
        """
        this function returns a new state used only for calculating the expected cost of future stochastic events
        the state returned includes only opened events, with new indexs which are fit for new run
        the new state only includes events and cars that are not commited
        :param :
        :return: dictionary of new states with opened not commited events, not commited cars and stochastic events
        """
        wantedState   = deepcopy(self)
        newEventsDict = {}
        newCarDict    = {}
        stochasticWantedStates = {}
        # find all events that are not commited and opened and add to fake state:
        for i,k in enumerate(self.events.getUnCommitedKeys()):
            tempEvent = self.events.getObject(k).createCopy()
            if tempEvent.status == Status.OPENED:
                newEvent = Event(tempEvent.position,i,tempEvent.startTime,tempEvent.endTime)
                newEventsDict[i] = newEvent
        # find all cars that are not commited and add to fake state:
        for i,k in enumerate(self.cars.getUnCommitedKeys()):
            tempCar = self.cars.getObject(k).createCopy()
            newCar  = SearchCar(tempCar.position,i)
            newCarDict[i] = newCar
        # create new monitor for state:
        newCars = commitMonitor({}, newCarDict)
        # add stochastic events to state and create number of states as number of stochastic runs
        for i in range(len(stochasticEventsDict)):
            # extract event position and time window from dictionary of stochastic events
            eventsPos = stochasticEventsDict[i]['eventsPos']
            eventsTimeWindow = stochasticEventsDict[i]['eventsTimeWindow']
            eventsDict = {}
            # create new uncommited event dict to add to already existed not commited event dict
            for j in range(eventsPos.shape[0]):
                eventsDict[j + len(newEventsDict)] = SearchEvent(eventsPos[j, :], len(newEventsDict) + j, eventsTimeWindow[j, 0],eventsTimeWindow[j, 1])
            newEventsDict.update(eventsDict)
            newEvents = commitMonitor({},newEventsDict)
            wantedState.events = newEvents
            wantedState.cars   = newCars
            # add updated state to dictionary of states for actual simulation
            stochasticWantedStates[i] = deepcopy(wantedState)
        return stochasticWantedStates


    def getDistanceMatrix(self):
        # create distance matrix
        carPositions = np.vstack([self.cars.getObject(i).position for i in range(self.cars.length())])
        evePositions = np.vstack([self.events.getObject(i).position for i in range(self.events.length())])
        matrix = cdist(carPositions, evePositions, metric="cityblock")
        return matrix

    def updateCost(self, counter, move):
        cost = 0
        cost += np.sum(move)
        cost += len(self.cars.commited)
        cost += counter[Status.OPENED]
        cost += counter[Status.CANCELED]*self.cancelPenalty
        cost -= counter[Status.CLOSED]*self.closeReward
        self.gval += cost
        return


def moveGenerator(numCars):
    moveIter = itertools.product([(0,0), (0,1), (0,-1), (-1,0), (1,0)], repeat=numCars)
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

def descendentGenerator(state):
    commitGen = commitGenerator(list(state.cars.getUnCommitedKeys()), list(state.events.getUnCommitedKeys()))
    for commit in commitGen:
        # commit cars
        newCommitState = deepcopy(state)
        # update root,parent and time for all descendants
        newCommitState.root     = False
        newCommitState.parent   = state
        newCommitState.time     += 1
        newCommitState.commitCars(commit)
        numUnCommitedCars = len(newCommitState.cars.notCommited)
        moveGen = moveGenerator(numUnCommitedCars)
        for possibleMove in moveGen:
            # create copy for new descendent
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
            yield newMoveState

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


def createStochasticEvents(numStochasticRuns ,gridSize,startTime,endTime,lam,eventsTimeWindow):
    """
    this function creates stochastic events with time and position and adds them to dictionary
    :param numStochasticRuns: number of stochastic runs to create
    :param startingId: the id from which to start adding events (num opened not commited events)
    :param gridSize: size of grid to add events
    :param startTime: time from which to start stochastic events
    :param endTime: last time for events
    :param lam: rate of events for poisson distribution
    :param eventsTimeWindow: time each event is opened
    :return: Dictionary of stochastic runs, in each stochastic run is the dictionary of events
    """
    stochasticEventsDict = {}
    for i in range(numStochasticRuns):
        eventPos, eventTimeWindow = createEventsDistribution(gridSize, startTime, endTime, lam, eventsTimeWindow)
        stochasticEventsDict[i] = {'eventsPos':eventPos,'eventsTimeWindow':eventTimeWindow}
    return stochasticEventsDict


def anticipatorySimulation(initState,nStochastic, gs, tPred, eTimeWindow, lam, shouldPrint=False):
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
    isGoal = False
    current = initState
    while not isGoal:
        currentTime = current.time
        optionalExpectedCost = []
        optionalStatesList   = []
        optionalTotalCost    = []
        stochasticEventsDict = createStochasticEvents(nStochastic, gs, 1, 1 + tPred,lam,eTimeWindow)
        for i,optionalState in enumerate(descendentGenerator(current)):
            optionalStatesList.append(optionalState)
            stochasticStatesDict = optionalState.getStateForStochasticRuns(stochasticEventsDict)
            # run determistic algorithm on stochastic state and keep the cost of the run
            stochasticCost = np.zeros(shape = (nStochastic,1))
            for j in range(len(stochasticEventsDict)):
                tempState    = stochasticStatesDict[j]
                stateForCalc = SearchState(root=True, carMonitor=tempState.cars, eventMonitor=tempState.events, cost=0,
                                        parent=None, time=0, weight=1, cancelPenalty=tempState.cancelPenalty,
                                        closeReward=tempState.closeReward, timeDelta=tempState.td, eps=tempState.eps)

                stime       = time.clock()
                p           = aStar(stateForCalc)
                etime       = time.clock()
                runTime     = etime - stime
                if shouldPrint:
                    print('finished stochastic run '+str(j) +'/'+str(len(stochasticEventsDict)))
                    print('run time for determistic is:'+str(round(runTime,3)))
                    print('cost of determinstic is: ' + str(p[-1].gval))
                stochasticCost[j] = p[-1].gval
            # calculate expected cost of all stochastic runs for this spesific optional State
            expectedCost = np.mean(stochasticCost)
            if shouldPrint:
                print('finished optional state # '+str(i))
                print('state cost is: ' + str(optionalState.gval) + ', expected cost is: '+str(expectedCost))
            optionalExpectedCost.append(expectedCost)
            optionalTotalCost.append(expectedCost+optionalState.gval)
        chosenTotalCost = np.min(np.array(optionalTotalCost))
        if shouldPrint:
            print('chosen cost is: '+str(chosenTotalCost))
        chosenIndex     = np.argmin(np.array(optionalTotalCost))
        chosenState     = optionalStatesList[chosenIndex]
        current         = chosenState
        # check if this state is a goal or not-
        if current.goalCheck():
            isGoal = True
            print('finished run - total cost is:'+str(current.gval))
    return current




def main():
    np.random.seed(10)
    shouldPrint = True
    # params
    epsilon     = 0.001  # distance between locations to be considered same location
    lam         = 10 / 30  # number of events per hour/ 60
    lengthSim   = 35  # minutes
    numStochasticRuns   = 10
    # initilize stochastic event list for each set checked -
    lengthPrediction    = 6
    deltaTimeForCommit  = 5
    closeReward         = 10
    cancelPenalty       = 50

    gridSize            = 4
    deltaOpenTime       = 5
    numCars             = 1

    # carPos              = np.array([0, 0]).reshape(2, numCars)
    # eventPos            = np.array([[5, 5], [5, 10]])
    # eventStartTime      = np.array([5, 10])
    # eventEndTime        = eventStartTime + deltaOpenTime

    carPos              = np.reshape(np.random.randint(0, gridSize, 2 * numCars), (2, numCars))

    eventPos,eventTimes = createEventsDistribution(gridSize, 0, lengthSim, lam, deltaOpenTime)
    eventStartTime      = eventTimes[:,0]
    eventEndTime        = eventTimes[:,1]



    uncommitedCarDict   = {}
    commitedCarDict     = {}
    uncommitedEventDict = {}
    commitedEventDict   = {}
    numEvents           = eventTimes.shape[0]
    for i in range(numCars):
        tempCar = Car(carPos[:, i], i)
        uncommitedCarDict[i] = tempCar
    for i in range(numEvents):
        tempEvent = Event(eventPos[i,:], i, eventStartTime[i], eventEndTime[i])
        uncommitedEventDict[i] = tempEvent

    carMonitor = commitMonitor(commitedCarDict, uncommitedCarDict)
    eventMonitor = commitMonitor(commitedEventDict, uncommitedEventDict)
    initState = State(root=True, carMonitor=carMonitor, eventMonitor=eventMonitor, cost=0, parent=None, time=0,
                      cancelPenalty=cancelPenalty, closeReward=closeReward,
                      timeDelta=deltaTimeForCommit,eps=epsilon)
    stime = time.clock()
    p = anticipatorySimulation(initState, numStochasticRuns, gridSize, lengthPrediction, deltaOpenTime, lam, shouldPrint=True)
    etime = time.clock()
    runTime = etime - stime
    print('cost is:' + str(p[-1].gval))
    print('run time is: ' + str(runTime))
    actualMaxTime       = len(p)
    numUnCommited       = np.zeros(actualMaxTime)
    numCommited         = np.zeros(actualMaxTime)
    timeVector          = list(range(actualMaxTime))
    closedEvents        = np.zeros(actualMaxTime)
    openedEvents        = np.zeros(actualMaxTime)
    canceledEvents      = np.zeros(actualMaxTime)
    allEvents           = np.zeros(actualMaxTime)
    for i, st in enumerate(p):
        numUnCommited[i]    = len(list(st.events.getUnCommitedKeys()))
        numCommited[i]      = len(list(st.events.getCommitedKeys()))
        for iE in range(numEvents):
            eventTemp       = st.events.getObject(iE)
            if eventTemp.status == Status.CANCELED:
                canceledEvents[i] += 1
            elif eventTemp.status == Status.CLOSED:
                closedEvents[i] += 1
            elif eventTemp.status == Status.OPENED:
                openedEvents[i] += 1
        allEvents[i] = canceledEvents[i] + closedEvents[i] + openedEvents[i]

    # dump logs
    with open('SimAnticipatoryCommitResults_' + str(1) + 'weight_' + str(numCars) + 'numCars_' + str(lam) + 'lam_' + str(
            gridSize) + 'gridSize.p', 'wb') as out:
        pickle.dump({'runTime'          : runTime,
                     'time'             : timeVector,
                     'OpenedEvents'     : openedEvents,
                     'closedEvents'     : closedEvents,
                     'canceledEvents'   : canceledEvents,
                     'allEvents'        : allEvents, 'cost': p[-1].gval}, out)
    imageList = []
    for s in p:
        imageList.append(plotForGif(s, numEvents, numCars, gridSize))
    imageio.mimsave('./gif_AStarSolution_' + str(gridSize) + 'grid_' + str(numCars) + 'cars_' + str(numEvents) + 'events_' + str(
            deltaOpenTime) + 'eventsTw_' + str(actualMaxTime) + 'maxTime.gif', imageList, fps=1)
    return


def plotForGif(s, ne, nc, gs):
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
        if eventTemp.status == Status.OPENED and eventTemp.commited:
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='b', alpha=0.7)
        elif eventTemp.status == Status.OPENED and eventTemp.commited == False:
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='b', marker='*', alpha=0.7)
        elif (eventTemp.status == Status.CLOSED):
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='g', alpha=0.2)
        elif (eventTemp.status == Status.CANCELED):
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='r', alpha=0.2)
        else:
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='y', alpha=0.2)
    ax.set_xlim([-1, gs + 1])
    ax.set_ylim([-1, gs + 1])
    ax.grid(True)
    plt.legend()
    # Used to return the plot as an image rray
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image







if __name__ == '__main__':
    main()
    print('Done.')
