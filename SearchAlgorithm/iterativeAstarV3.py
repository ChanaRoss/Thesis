from copy import deepcopy
from enum import Enum
import itertools
import numpy as np
import heapq, time
from scipy.spatial.distance import cdist
import  pickle
from matplotlib import pyplot as plt
import imageio

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
        self.position = np.reshape(position, (2,))
        self.id = id
        self.commited = False
        self.targetId = None
        self.path = [self.position]
        return

    def __lt__(self, other):
        if not isinstance(other, Car):
            raise Exception("other must be of type Car.")
        else:
            return self.id<=other.id

    def __eq__(self, other):
        if not isinstance(other, Car):
            raise Exception("other must be of type Car.")
        elif self.id==other.id:
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
        self.id = id
        self.position  = np.reshape(position, (2,))
        self.commited  = False
        self.startTime = start
        self.endTime   = end
        self.status    = Status.PREOPENED
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

    def updateStatus(self):
        prev = self.status
        if self.status == Status.CLOSED or self.status == Status.CANCELED:
            pass
        elif self.startTime<=0 and self.endTime>=0 and self.commited:
            # event is opened and commited to
            self.status = Status.OPENED_COMMITED
        elif self.startTime<=0 and self.endTime>=0 and not self.commited:
            self.status = Status.OPENED_NOT_COMMITED
        elif self.endTime<0:
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

    def getCommitedKeys(self):
        return self.commited.keys()

    def getUnCommitedKeys(self):
        return self.notCommited.keys()

    def length(self):
        return len(self.commited)+len(self.notCommited)

class SearchState:
    def __init__(self,root , carMonitor, eventMonitor, cost, parent, time, weight=1, openedCommitedPenalty=1,openedNotCommitedPenalty = 5, cancelPenalty=100, closeReward=50, timeDelta=5, eps=0.001):
        self.cars                       = carMonitor
        self.events                     = eventMonitor
        self.gval                       = cost
        self.hval                       = float('Inf')
        self.weight                     = weight
        self.openedCommitedPenalty      = openedCommitedPenalty
        self.openedNotCommitedPenalty   = openedNotCommitedPenalty
        self.cancelPenalty              = cancelPenalty
        self.closeReward                = closeReward
        self.parent                     = parent
        self.root                       = root
        self.td                         = timeDelta
        self.time                       = time
        self.eps                        = eps
        return

    def __lt__(self, other):
        if not isinstance(other, SearchState):
            raise Exception("States must be compared to another State object.")
        else:
            return self.getFval()<=other.getFval()

    def __eq__(self, other):
        if not isinstance(other, SearchState):
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

    def getFval(self):
        return self.gval+self.hval*self.weight

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
            opened = self.events.getObject(k).status == Status.OPENED_COMMITED
            pre    = self.events.getObject(k).status == Status.PREOPENED
            if opened or pre:
                return False
        # check uncommited events
        for k in self.events.getUnCommitedKeys():
            opened = self.events.getObject(k).status == Status.OPENED_NOT_COMMITED
            pre    = self.events.getObject(k).status == Status.PREOPENED
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
            delta = targetPosition - tempCar.position
            if delta[0]!=0:
                tempCar.position[0] += np.sign(delta[0])
            else:
                tempCar.position[1] += np.sign(delta[1])
        return

    def updateStatus(self, matrix):
        # update event status
        counter = {Status.OPENED_COMMITED : 0 , Status.OPENED_NOT_COMMITED : 0, Status.CLOSED: 0, Status.CANCELED: 0}
        for eventId in range(self.events.length()):
            tempEvent = self.events.getObject(eventId)
            tempEvent.startTime -= 1
            tempEvent.endTime -= 1
            newStatus = tempEvent.updateStatus()
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

    def updateHeuristic(self, matrix):
        # commited events
        h1 = 0
        for carId in self.cars.getCommitedKeys():
            tempCar = self.cars.getObject(carId)
            closeTime = self.events.getObject(tempCar.targetId).endTime
            dist = matrix[carId, tempCar.targetId]
            if dist<=closeTime:
                h1 -= self.closeReward
            else:
                h1 += self.cancelPenalty
        # uncommited events
        h2 = 0
        for eventId in self.events.getUnCommitedKeys():
            tempEvent = self.events.getObject(eventId)
            startTime = tempEvent.startTime
            closeTime = tempEvent.endTime
            minDist = np.min(matrix[:, eventId])
            if minDist-closeTime>0:
                h2 += self.cancelPenalty
            else:
                h2 += np.max([minDist-startTime, 0]) + minDist - self.closeReward
        self.hval = h1 + h2
        return


class Heap:
    def __init__(self):
        self.array = list()

    def __len__(self):
        return len(self.array)

    def insert(self, obj):
        heapq.heappush(self.array, obj)
        return

    def extractMin(self):
        return heapq.heappop(self.array)  # get minimum from heap

    def empty(self):
        return len(self.array)==0

    def heapify(self):
        heapq.heapify(self.array)
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
            # update heuristic: heuristic of new state
            newMoveState.updateHeuristic(dm)
            # yield the new state
            yield newMoveState

def aStar(initState, shouldPrint=False):
    # start timers
    t0 = time.clock()
    calcTime = 0
    objectCreationTime = 0
    # create heap
    openHeap = Heap()
    closed = {}
    # insert initial state
    openHeap.insert(initState)
    cnt = 0
    while not openHeap.empty():
        current = openHeap.extractMin()
        if (current not in closed) or (current in closed and closed[current]>current.gval):
            # remove previous equivalent state, and enter the new one
            closed[current] = current.gval
            # check if current state is a goal state, return path if it is
            if current.goalCheck():
                if shouldPrint:
                    print("solution found. Total Time: {0}, Total States explored: {1}".format(round(time.clock()-t0, 4), len(closed) + len(openHeap)))
                    print("rep count: {0}, calc time: {1}, object creation time: {2}".format(cnt, calcTime, objectCreationTime))
                return current.path()
            # calculate all possible following states
            for descendent in descendentGenerator(current):
                # check if is in closed
                if (descendent in closed) and (descendent.gval<closed[descendent]):
                    closed.pop(descendent)  # remove from closed
                    openHeap.insert(descendent)  # add to open
                else:
                    openHeap.insert(descendent)  # fresh state, insert into the existing heap

    if shouldPrint:
        print("solution not found. Total Time: {0}, Total States explored: {1}".format(round(time.clock()-t0,4), len(closed) + len(openHeap)))
        print("rep count: {0}, calc time: {1}, object creation time: {2}".format(cnt, calcTime, objectCreationTime))
    return None  # no solution exists



def main():
    np.random.seed(10)
    deltaTimeForCommit          = 5
    closeReward                 = 50
    cancelPenalty               = 100
    openedCommitedPenalty       = 1
    openedNotCommitedPenalty    = 10

    gridSize      = 5
    deltaOpenTime = 4
    numCars       = 1
    numEvents     = 5

    maxTime       = 10

    carPos         = np.reshape(np.random.randint(0, gridSize, 2 * numCars), (2,numCars))
    eventPos       = np.reshape(np.random.randint(0, gridSize, 2 * numEvents), ( 2,numEvents))
    eventStartTime = np.random.randint(0, maxTime, numEvents)
    eventEndTime   = deltaOpenTime + eventStartTime

    # carPos              = np.array([0,0]).reshape(2,numCars)
    # eventPos            = np.array([[5,5],[5,10]])
    # eventStartTime      = np.array([2,10])
    # eventEndTime        = eventStartTime + deltaOpenTime
    uncommitedCarDict   = {}
    commitedCarDict     = {}
    uncommitedEventDict = {}
    commitedEventDict   = {}
    for i in range(numCars):
        tempCar = Car(carPos[:,i],i)
        uncommitedCarDict[i] = tempCar
    for i in range(numEvents):
        tempEvent = Event(eventPos[:,i],i,eventStartTime[i],eventEndTime[i])
        uncommitedEventDict[i] = tempEvent

    carMonitor   = commitMonitor(commitedCarDict,uncommitedCarDict)
    eventMonitor = commitMonitor(commitedEventDict,uncommitedEventDict)
    initState    = SearchState(root = True,carMonitor= carMonitor,eventMonitor = eventMonitor,cost = 0,parent = None,time = 0,weight =1,
                               cancelPenalty=cancelPenalty,closeReward=closeReward, openedCommitedPenalty=openedCommitedPenalty
                               ,openedNotCommitedPenalty = openedNotCommitedPenalty,timeDelta=deltaTimeForCommit,eps = 0.001)
    stime   = time.clock()
    p       = aStar(initState)
    etime   = time.clock()
    runTime = etime - stime
    print('cost is:' + str(p[-1].gval))
    print('run time is: '+str(runTime))
    actualMaxTime = len(p)
    numUnCommited = np.zeros(actualMaxTime)
    numCommited   = np.zeros(actualMaxTime)
    timeVector    = list(range(actualMaxTime))
    closedEvents  = np.zeros(actualMaxTime)
    openedEvents  = np.zeros(actualMaxTime)
    canceledEvents= np.zeros(actualMaxTime)
    allEvents     = np.zeros(actualMaxTime)
    for i,st in enumerate(p):
        numUnCommited[i]   = len(list(st.events.getUnCommitedKeys()))
        numCommited[i]     = len(list(st.events.getCommitedKeys()))
        for iE in range(numEvents):
            eventTemp = st.events.getObject(iE)
            if eventTemp.status == Status.CANCELED:
                canceledEvents[i] += 1
            elif eventTemp.status == Status.CLOSED:
                closedEvents[i]   += 1
            elif eventTemp.status == Status.OPENED_COMMITED or eventTemp.status == Status.OPENED_NOT_COMMITED:
                openedEvents[i]   += 1
        allEvents[i] = canceledEvents[i]+closedEvents[i]+openedEvents[i]


    # dump logs
    with open( 'MyAStarResult_' + str(1) + 'weight_' + str(numCars) + 'numCars_' + str(numEvents) + 'numEvents_' + str(gridSize) + 'gridSize.p', 'wb') as out:
        pickle.dump({'runTime': runTime,
                     'time': timeVector,
                     'solution':p[-1],
                     'OpenedEvents'  : openedEvents,
                     'closedEvents'  : closedEvents,
                     'canceledEvents': canceledEvents,
                     'allEvents'     : allEvents, 'cost': p[-1].gval}, out,pickle.HIGHEST_PROTOCOL)
    imageList = []
    for s in p:
        imageList.append(plotForGif(s, numEvents,numCars, gridSize))
    imageio.mimsave('./gif_AStarSolution_' + str(gridSize) + 'grid_' + str(numCars) + 'cars_' + str(numEvents) + 'events_' + str(deltaOpenTime) + 'eventsTw_' + str(maxTime) + 'maxTime.gif', imageList, fps=1)


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
