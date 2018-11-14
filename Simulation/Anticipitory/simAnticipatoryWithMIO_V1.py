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
# for graphics
import seaborn as sns
from matplotlib import pyplot as plt
import imageio
sns.set()
# my files
sys.path.insert(0, '/home/chana/Documents/Thesis/FromGitFiles/MixedIntegerOptimization/')
from offlineOptimizationProblem_TimeWindow import runMaxFlowOpt


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
    def __init__(self,root , carMonitor, eventMonitor, cost, parent, time,openedCommitedPenalty = 1,openedNotCommitedPenalty=5, cancelPenalty=50, closeReward=10, timeDelta=5, eps=0.001):
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
            if delta[0]!=0:
                tempCar.position[0] += np.sign(delta[0])
            else:
                tempCar.position[1] += np.sign(delta[1])
        return

    def updateStatus(self, matrix):
        # update event status
        counter = {Status.OPENED_COMMITED : 0,Status.OPENED_NOT_COMMITED : 0, Status.CLOSED: 0, Status.CANCELED: 0}
        for eventId in range(self.events.length()):
            tempEvent               = self.events.getObject(eventId)
            newStatus               = tempEvent.updateStatus(self.time)
            if newStatus is not None:
                counter[newStatus] += 1

        # update commited cars and events
        for carId in range(self.cars.length()):
            tempCar = self.cars.getObject(carId)
            if tempCar.commited:
                if matrix[carId, tempCar.targetId]<=self.eps and self.events.getObject(tempCar.targetId).status==Status.OPENED_COMMITED:
                    self.events.getObject(tempCar.targetId).status = Status.CLOSED  # update event status
                    self.events.unCommitObject(tempCar.targetId)  # uncommit event
                    self.cars.unCommitObject(carId)  # uncommit car
                    tempCar.uncommit()  # update car field
                    counter[Status.CLOSED] += 1
            else:  # update uncommited events
                closedEvents = np.where(matrix[carId, :]<=self.eps)
                for e in closedEvents[0]:
                    tempEvent = self.events.getObject(e)
                    if tempEvent.status==Status.OPENED_NOT_COMMITED:  # uncommited event reached
                        tempEvent.status = Status.CLOSED  # close event
                        counter[Status.CLOSED] += 1  # incriment counter
        return counter


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
        cost += counter[Status.OPENED_COMMITED]*self.openedCommitedPenalty
        cost += counter[Status.OPENED_NOT_COMMITED]*self.openedNotCommitedPenalty
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
    else:
        yield([])

def descendentGenerator(state):
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


def anticipatorySimulation(initState,nStochastic, gs, tPred, eTimeWindow, lam, aStarWeight = 1,shouldPrint=False):
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
            # initilize variables for stochastic optimization
            carsPos        = np.zeros(shape = (optionalState.cars.length(),2))
            currentEventsPos      = []
            currentEventStartTime = []
            currentEventsEndTime   = []
            stochasticCost = np.zeros(shape=(nStochastic, 1))
            # get car locations from state -
            for d,k in enumerate(optionalState.cars.getUnCommitedKeys()):
                carsPos[d,:] = deepcopy(optionalState.cars.getObject(k).position)
            # get opened event locations from state -
            for k in optionalState.events.getUnCommitedKeys():
                if optionalState.events.getObject(k).status == Status.OPENED_NOT_COMMITED:
                    currentEventsPos.append(deepcopy(optionalState.events.getObject(k).position))
                    currentEventStartTime.append(deepcopy(optionalState.events.getObject(k).startTime))
                    currentEventsEndTime.append(deepcopy(optionalState.events.getObject(k).endTime))
            # run deterministic optimization for stochastic events -
            for j in range(len(stochasticEventsDict)):
                if len(stochasticEventsDict[j]['eventsPos']) + len(currentEventsPos) > 0:
                    # there are events to be tested in determinstic optimization:
                    eventsPos = deepcopy(currentEventsPos)
                    eventsStartTime = deepcopy(currentEventStartTime)
                    eventsEndTime = deepcopy(currentEventsEndTime)
                    temp = [eventsPos.append(e) for e in stochasticEventsDict[j]['eventsPos']]
                    temp = [eventsStartTime.append(e[0]) for e in stochasticEventsDict[j]['eventsTimeWindow']]
                    temp = [eventsEndTime.append(e[1]) for e in stochasticEventsDict[j]['eventsTimeWindow']]
                    eventsPos        = np.array(eventsPos)
                    eventsStartTime  = np.array(eventsStartTime)
                    eventsEndTime    = np.array(eventsEndTime)
                    stime = time.clock()
                    m,obj = runMaxFlowOpt(0, carsPos, eventsPos, eventsStartTime, eventsEndTime, optionalState.closeReward, optionalState.cancelPenalty, optionalState.openedNotCommitedPenalty,0)
                    etime = time.clock()
                    runTime = etime - stime
                    # if shouldPrint:
                    #     print("finished MIO for run:"+str(j+1)+"/"+str(len(stochasticEventsDict)))
                    #     print("run time of MIO is:"+str(runTime))
                    #     print("cost of MIO is:"+str(-obj.getValue()))
                    stochasticCost[j] = -obj.getValue()
                else:
                    stochasticCost[j] = 0
            # calculate expected cost of all stochastic runs for this spesific optional State
            expectedCost = np.mean(stochasticCost)
            if shouldPrint:
                print('finished optional state # '+str(i))
                print('state cost is: ' + str(optionalState.gval) + ', expected cost is: '+str(expectedCost) +' , commited cost is:'+str(optionalState.optionalGval))
            optionalExpectedCost.append(expectedCost)
            # optional total cost includes the actual cost of movement + optional cost for commited events + expected cost of future events
            optionalTotalCost.append(expectedCost+optionalState.gval+ optionalState.optionalGval)
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
        # dump logs
        with open('SimAnticipatoryCommitResults_' + str(1) + 'weight_' + str(current.cars.length()) + 'numCars_' + str(lam) + 'lam_' + str(gs) + 'gridSize.p', 'wb') as out:
            pickle.dump({'time': currentTime,
                         'currentState': current,
                         'cost': current.gval}, out)
    return current.path()




def main():
    np.random.seed(10)
    shouldPrint = True
    # params
    epsilon     = 0.001  # distance between locations to be considered same location
    lam         = 30 / 40  # number of events per hour/ 60
    lengthSim   = 15  # minutes
    numStochasticRuns   = 100
    # initilize stochastic event list for each set checked -
    lengthPrediction    = 4
    deltaTimeForCommit  = 10
    closeReward         = 50
    cancelPenalty       = 100
    openedCommitedPenalty    = 1
    openedNotCommitedPenalty = 5

    gridSize            = 7
    deltaOpenTime       = 5
    numCars             = 2

    aStarWeight         = 10
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
    numEvents           = eventStartTime.shape[0]
    for i in range(numCars):
        tempCar = Car(carPos[:, i], i)
        uncommitedCarDict[i] = tempCar
    for i in range(numEvents):
        tempEvent = Event(eventPos[i,:], i, eventStartTime[i], eventEndTime[i])
        uncommitedEventDict[i] = tempEvent

    carMonitor = commitMonitor(commitedCarDict, uncommitedCarDict)
    eventMonitor = commitMonitor(commitedEventDict, uncommitedEventDict)
    initState = State(root=True, carMonitor=carMonitor, eventMonitor=eventMonitor, cost=0, parent=None, time=0,
                      openedNotCommitedPenalty = openedNotCommitedPenalty, openedCommitedPenalty = openedCommitedPenalty,
                      cancelPenalty=cancelPenalty, closeReward=closeReward,
                      timeDelta=deltaTimeForCommit,eps=epsilon)
    stime = time.clock()
    p = anticipatorySimulation(initState, numStochasticRuns, gridSize, lengthPrediction, deltaOpenTime, lam,aStarWeight = aStarWeight, shouldPrint=True)
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
            eventTemp = st.events.getObject(iE)
            if eventTemp.status == Status.CANCELED:
                canceledEvents[i] += 1
            elif eventTemp.status == Status.CLOSED:
                closedEvents[i]   += 1
            elif eventTemp.status == Status.OPENED_COMMITED or eventTemp.status == Status.OPENED_NOT_COMMITED:
                openedEvents[i]   += 1
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


def plotForGif(s, ne,nc, gs):
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
