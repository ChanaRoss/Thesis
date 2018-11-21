from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist
import time
import itertools
import imageio
import heapq
import pickle
sns.set()


__author__ = "Chana Ross, Yoel Ross"
__copyright__ = "Copyright 2018"
__credits__ = ["Yoel Ross", "Tamir Hazan", "Erez Karpas"]
__version__ = "3.0"
__maintainer__ = "Chana Ross"
__email__ = "schanaby@campus.technion.ac.il"
__status__ = "Thesis"


# Class definitions
class SearchState:
    def __init__(self, carPos, eventPos, eventTimes ,eventCloseTimes, eventReward, eventPenalty, eventsCanceled, eventsAnswered, heuristicVal, costVal, parent,hWeight):
        self.carPos                 = carPos
        self.hWeight                = hWeight
        self.eventPos               = eventPos
        self.eventTimes             = eventTimes
        self.eventCloseTimes        = eventCloseTimes
        self.eventReward            = eventReward
        self.eventPenalty           = eventPenalty
        self.eventsCanceled         = eventsCanceled   # true if the event is opened, false if event closed or canceled. starts as true
        self.eventsAnswered         = eventsAnswered  # true if event is answered , false if event is canceled. starts as false
        self.time                   = parent.time+1 if parent is not None else 0  # time is one step ahead of parent
        self.hval                   = heuristicVal
        self.gval                   = costVal
        self.parent                 = parent  # predecessor in graph
        self.root                   = parent is None  # true of state is the root, false otherwise
        return

    def __lt__(self, other):
        # make sure comparison is to SearchState object
        try:
            assert (isinstance(other, SearchState))
        except:
            raise TypeError("must compare to SearchState object.")
        # return lt check
        return self.getFval() < other.getFval()

    def __eq__(self, other):
        # make sure comparison is to SearchState object
        try:
            assert(isinstance(other, SearchState))
        except:
            raise TypeError("must compare to SearchState object.")
        # check
        carEq = np.array_equal(self.carPos, other.carPos)
        eveEq = np.array_equal(self.eventPos,other.eventPos)
        etmEq = np.array_equal(self.eventTimes,other.eventTimes)
        etcEq = np.array_equal(self.eventCloseTimes,other.eventCloseTimes)
        sttEq = np.array_equal(self.eventsCanceled, other.eventsCanceled)
        astEq = np.array_equal(self.eventsAnswered,other.eventsAnswered)
        timEq = self.time == other.time
        return carEq and eveEq and etcEq and etmEq and sttEq and astEq and timEq

    def __repr__(self):
        return "time: {0}, cost: {1}, heuristic: {2}, root: {3}, goal: {4}\n".format(self.time,
                                                                                     self.gval,
                                                                                     self.hval,
                                                                                     self.root,
                                                                                     self.goalCheck())

    def __hash__(self):
        carPosVec  = np.reshape(self.carPos, self.carPos.size)
        evePosVec  = np.reshape(self.eventPos, self.eventPos.size)
        eveSttVec  = self.eventsCanceled.astype(np.int32)
        eveAnStVec = self.eventsAnswered.astype(np.int32)
        stateTime  = np.reshape(np.array(self.time), 1)
        hv = np.hstack([carPosVec , evePosVec , eveSttVec ,eveAnStVec , stateTime])
        return hash(tuple(hv))

    def goalCheck(self):
        # if all events have been answered or canceled, we have reached the goal
        return np.sum(np.logical_or(self.eventsAnswered,self.eventsCanceled)) == self.eventsAnswered.size

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

    def getFval(self):
        return self.hval*self.hWeight + self.gval


class Heap():
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


# Function definitions
def createMoveMatrix(numCars):
    """
    returns a 3D array:
    mat[i][j][0] = delta of car j in possible move i in x axis
    mat[i][j][1] = delta of car j in possible move i in y axis
    :param numCars: number of cars in scenario
    :return: numpy integer array (int8)
    """
    return np.array(list(itertools.product([(0,0), (0,1), (0,-1), (-1,0), (1,0)], repeat=numCars))).astype(np.int8)

def calcDistanceMatrix(possibleCarPos, eventPos):
    distanceStack = np.zeros(shape=(possibleCarPos.shape[0], possibleCarPos.shape[1], eventPos.shape[0]))
    for i in range(possibleCarPos.shape[0]):
        distanceStack[i, :, :] = cdist(possibleCarPos[i,:,:], eventPos, metric='cityblock')
    return distanceStack.astype(np.int32)

def calcNewCarPos(carPos, moveMat):
    """
    calculate all the new car position possibilities
    :param carPos: 2D matrix of car positions shape=(nc,2)
    :param moveMat: 3D matrix of possible moves shape=(5**nc,nc,2)
    :return: 3D matrix of all possible positions, shape=(5**nc,nc,2) (int32)
    """
    tiledCarPos = np.tile(carPos, reps=(moveMat.shape[0], 1, 1))
    newCarPos = tiledCarPos + moveMat
    return newCarPos.astype(np.int32)

def distCost(moveMatrix, eventsAnswered,eventsCanceled, possibleNewEventsAnswered,possibleNewEventsCanceled, newEventTimes,eventReward,eventPenalty):
    """
    calculates cost
    assume the cost is the sum of :
    1. distance each car is moving in this time step
    2. time each event is waiting
    3. discount the profit of an event that is being closed
    :param moveMatrix:
    :param newEventTimes: time vector of opening times of events (updated)
    :param possibleNewEventStatus: all possible event status for next move
    :return: cost for each possible move, index of cost matches the move matrix rows (int32)
    """
    distComponent = np.sum(np.abs(moveMatrix), axis=(1,2))
    possibleNewEventTimes = np.tile(newEventTimes, (moveMatrix.shape[0],1))
    # check which events are not canceled and not answered
    possibleNewEventsOpened = np.logical_and(np.logical_not(possibleNewEventsAnswered),np.logical_not(possibleNewEventsCanceled))
    openEventsComponent     = np.sum(np.logical_and(possibleNewEventTimes<=0, possibleNewEventsOpened), axis=1).astype(np.int32)
    previousEventsAnswered  = np.tile(eventsAnswered, reps=(possibleNewEventsOpened.shape[0], 1))
    previousEventsCanceled  = np.tile(eventsCanceled, reps=(possibleNewEventsOpened.shape[0], 1))
    closingEventComponent   = eventReward*np.sum(np.logical_xor(possibleNewEventsAnswered, previousEventsAnswered), axis=1)
    canceledEventsComponent = eventPenalty*np.sum(np.logical_xor(possibleNewEventsCanceled,previousEventsCanceled),axis = 1)
    # calculate cost with all components and return
    return (distComponent+openEventsComponent-closingEventComponent+canceledEventsComponent).astype(np.int32)

def heuristic(distanceMatrix, newEventTimes,newEventClosedTimes, possibleNewEventsAnswered,possibleNewEventsCanceled,eventReward,eventPenalty):
    """
    calculates heuristic value for each possible move
    :param distanceMatrix: 3D array including a 2d distance matrix for each possible move to all events
    :param eventTime: time of events starting
    :param eventStatus: boolian of open events
    :return:
    """
    # calc minimum ditances and costs
    minDists = np.min(distanceMatrix, axis=1)
    possibleEventCosts = minDists-np.tile(newEventTimes, reps=(minDists.shape[0],1))
    possibleNewEventsOpened = np.logical_and(np.logical_not(possibleNewEventsAnswered),np.logical_not(possibleNewEventsCanceled))
    isReachable =  (minDists - np.tile(newEventClosedTimes, reps=(minDists.shape[0],1)))<0
    possibleNewEventsReachable = np.logical_and(isReachable,possibleNewEventsOpened)
    possibleNewEventsNotReachable = np.logical_and(np.logical_not(isReachable),possibleNewEventsOpened)
    # filter out closed events for each move
    filteredPossibleEventCosts = np.multiply(possibleEventCosts, possibleNewEventsReachable.astype(np.int32))
    filteredMinDist = np.multiply(minDists, possibleNewEventsReachable)
    notReachablePenalty = eventPenalty*np.sum(possibleNewEventsNotReachable, axis =1)
    RechableReward      = eventReward*np.sum(possibleNewEventsReachable , axis = 1)
    # final event costs
    eventsCosts = np.maximum(filteredPossibleEventCosts, np.zeros_like(filteredPossibleEventCosts))
    finalCosts = np.sum((eventsCosts + filteredMinDist).astype(np.int32), axis=1) - RechableReward + notReachablePenalty
    return finalCosts

def calcNewEventStatus(distanceMatrix, eventsCanceled,eventsAnswered, newEventTimes , newEventClosedTimes, epsilon=0.001):
    """
    calculates the new status vectors
    :param distanceMatrix: matrix of distance between each event and car
    :param eventsCanceled: vector of canceled events status. true = event canceled, false = event opened/ answered
    :param eventsAnswered: vector of answered events status. true = event answered, false = event opened/ canceled
    :param newEventTime: vector of new events time (positive if event not opened yet)
    :param newEventClosedTime: vector of new events closing time (positive if event can be answered)
    :param epsilon:
    :return: canceled event status, answered event status
    """
    # convert distance matrix to boolean of approx zero (picked up events)
    step1 = np.sum((distanceMatrix <= epsilon), axis=1) >= 1
    eventsOpened = np.logical_and(np.logical_not(eventsAnswered),np.logical_not(eventsCanceled))
    # condition on event being open
    step2 = np.logical_and(step1, np.tile(eventsOpened, reps=(distanceMatrix.shape[0], 1)))
    # condition on event started (time<=0)
    step3 = np.logical_and(step2, np.tile(np.logical_and(newEventTimes <= 0,newEventClosedTimes>0) , reps=(distanceMatrix.shape[0], 1)))
    # new possible events answered status
    newPossibleEventsAnswered = np.tile(eventsAnswered, reps=(distanceMatrix.shape[0], 1))
    newPossibleEventsAnswered[step3] = 1
    newPossibleEventsAnswered.astype(np.bool_)
    # new possible events canceled status
    step2Closed = np.tile(np.logical_and(eventsOpened,newEventClosedTimes<=0),reps=(distanceMatrix.shape[0],1))
    newPossibleEventsCanceled = np.tile(eventsCanceled, reps=(distanceMatrix.shape[0], 1))
    newPossibleEventsCanceled[step2Closed] = 1
    newPossibleEventsCanceled.astype(np.bool_)
    return newPossibleEventsCanceled,newPossibleEventsAnswered

def insertOpen(state, heap, heapLookup):
    heap.insert(state)
    heapLookup[state] = state
    return

def aStar(initState, epsilon=0.001,shouldPrint=False):
    # start timers
    t0 = time.clock()
    calcTime = 0
    objectCreationTime = 0
    # create heap
    openHeap = Heap()
    closed = {}
    # insert initial state
    openHeap.insert(initState)
    # create move matrix
    nc = initState.carPos.shape[0]  # number of cars in problem
    movesMat = createMoveMatrix(nc)
    eventReward = initState.eventReward
    eventPenalty = initState.eventPenalty
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
            tc0 = time.clock()
            possibleNewCarPositions = calcNewCarPos(current.carPos, movesMat)
            distanceMat = calcDistanceMatrix(possibleNewCarPositions, current.eventPos)
            newEventTimes = current.eventTimes - 1  # create new time vector
            newEventClosedTimes = current.eventCloseTimes -1 # create new close time vector
            possibleNewEventsCanceled,possibleNewEventsAnswered = calcNewEventStatus(distanceMat, current.eventsCanceled,current.eventsAnswered, newEventTimes, newEventClosedTimes, epsilon)
            nextCosts = distCost(movesMat,current.eventsAnswered,current.eventsCanceled,possibleNewEventsAnswered,possibleNewEventsCanceled, newEventTimes,eventReward,eventPenalty) + current.gval  # costs of new frontier states
            nextHeuri = heuristic(distanceMat, newEventTimes, newEventClosedTimes,possibleNewEventsAnswered,possibleNewEventsCanceled, eventReward,eventPenalty)  # heuristic of new frontier states
            tc1 = time.clock()
            calcTime += tc1-tc0
            # create children states and add to open
            oc0 = time.clock()
            for i in range(possibleNewCarPositions.shape[0]):
                # create new state
                tempState = SearchState(possibleNewCarPositions[i,:],
                                        np.copy(current.eventPos),
                                        np.copy(newEventTimes),
                                        np.copy(newEventClosedTimes),
                                        eventReward,
                                        eventPenalty,
                                        possibleNewEventsCanceled[i,:],
                                        possibleNewEventsAnswered[i,:],
                                        nextHeuri[i],
                                        nextCosts[i],
                                        current,
                                        current.hWeight)
                # check if is in closed
                if (tempState in closed) and (tempState.gval<closed[tempState]):
                    closed.pop(tempState)  # remove from closed
                    openHeap.insert(tempState)  # add to open
                else:
                    openHeap.insert(tempState)  # fresh state, insert into the existing heap
            oc1 = time.clock()
            objectCreationTime += oc1-oc0
    if shouldPrint:
        print("solution not found. Total Time: {0}, Total States explored: {1}".format(round(time.clock()-t0,4), len(closed) + len(openHeap)))
        print("rep count: {0}, calc time: {1}, object creation time: {2}".format(cnt, calcTime, objectCreationTime))
    return None  # no solution exists



def main():
    np.random.seed(1)
    nc = 2
    ne = 15
    gs = 5
    maxTime = 15
    hWeight = 1
    eventTimeWindow = 5
    fromPickle = 1
    eventReward = 10
    eventPenalty = 100

    print("num cars: {0}, num events: {1}, grid size: {2}:{3}".format(nc, ne, gs, gs))
    if fromPickle:
        pickleName = 'logAnticipatory_10EventReward_8grid_2cars_35simLengh_20StochasticLength_5Prediction_5aStarWeight'
        lg = pickle.load(open('/home/chana/Documents/Thesis/FromGitFiles/Simulation/Anticipitory/Results/' + pickleName + '.p','rb'))
        eventDataTuple = [(e['timeStart'], e['position'],e['timeEnd']) for e in lg['events'].values()]
        carPos = np.vstack([np.array(c['position']) for c in lg['carDict'].values()])
        eveTim = np.zeros(len(eventDataTuple))
        eveCloseTime = np.zeros(len(eventDataTuple))
        evePos = np.zeros(shape=(len(eveTim), 2))
        eventsAnswered = np.zeros_like(eveTim).astype(np.bool_)
        eventsCancled = np.zeros_like(eveTim).astype(np.bool_)
        for i, et in enumerate(eventDataTuple):
            eveTim[i] = eventDataTuple[i][0]
            evePos[i, :] = np.array(eventDataTuple[i][1])
            eveCloseTime[i] = np.array(eventDataTuple[i][2])
    else:
        carPos = np.reshape(np.random.randint(0, gs, 2 * nc), (nc, 2))
        evePos = np.reshape(np.random.randint(0, gs, 2 * ne), (ne, 2))
        eveTim = np.random.randint(0, maxTime, ne)
        eveCloseTime = eventTimeWindow*np.ones_like(eveTim)+eveTim
        eventsAnswered = np.zeros_like(eveTim).astype(np.bool_)
        eventsCancled = np.zeros_like(eveTim).astype(np.bool_)
    initState = SearchState(carPos, evePos, eveTim,eveCloseTime,eventReward,eventPenalty, eventsCancled,eventsAnswered, float('inf'), 0, None, hWeight)
    stime = time.clock()
    p = aStar(initState)
    etime = time.clock()
    runTime = etime - stime
    print('cost is:' + str(p[-1].gval))
    openedEvents = [len([(s,e) for s,e in zip(a.eventTimes,a.eventCloseTimes) if (s<=0 and e>=0)]) for a in p]
    allEvents = [len([s for s in a.eventTimes if s<=0 ]) for a in p]
    timeVector = [a.time for a in p]
    closedEvents =[len([(e,c) for e,c in zip(a.eventsAnswered,a.eventsCanceled) if e or c]) for a in p]
    answeredEvents = [len([e for e in a.eventsAnswered if e]) for a in p]
    canceledEvents = [len([e for e in a.eventsCanceled if e]) for a in p]
    # dump logs
    with open(
            'MyAStarResult_' + str(hWeight) + 'weight_' + str(len(carPos)) + 'numCars_' + str(ne) + 'numEvents_' + str(
                    gs) + 'gridSize.p', 'wb') as out:
        pickle.dump({'runTime'          : runTime,
                     'time'             : timeVector,
                     'closedEvents'     : closedEvents,
                     'OpenedEvents'     : openedEvents,
                     'AnsweredEvents'   : answeredEvents,
                     'CanceledEvents'   : canceledEvents,
                     'AllEvents'        : allEvents,'cost':p[-1].gval}, out)
    imageList = []
    for s in p:
        imageList.append(plotForGif(s, ne, gs))
    imageio.mimsave('./gif_AStarSolution_' + str(gs) + 'grid_' + str(nc) + 'cars_' + str(ne) + 'events_'+str(eventTimeWindow)+'eventsTw_'+str(maxTime)+'maxTime.gif', imageList, fps=1)


def plotForGif(s, ne, gs):
    """
        plot cars as red points, events as blue points,
        and lines connecting cars to their targets
        :param carDict:
        :param eventDict:
        :return: image for gif
        """
    fig, ax = plt.subplots()
    ax.set_title('time: {0}'.format(s.time))
    ax.scatter(s.carPos[:, 0]  , s.carPos[:, 1], c = 'k', alpha=0.5)
    for i in range(ne):
        if (s.eventTimes[i] <= 0) and (s.eventCloseTimes[i]>=0 and (not s.eventsAnswered[i]) and (not s.eventsCanceled[i])):
            ax.scatter(s.eventPos[i, 0], s.eventPos[i, 1], c = 'b', alpha = 0.7)
        elif (s.eventsAnswered[i]):
            ax.scatter(s.eventPos[i,0],  s.eventPos[i,1],  c = 'g', alpha = 0.2)
        elif (s.eventsCanceled[i]):
            ax.scatter(s.eventPos[i, 0], s.eventPos[i, 1], c = 'r', alpha = 0.2)
        else:
            ax.scatter(s.eventPos[i, 0], s.eventPos[i, 1], c = 'y', alpha = 0.2)

    ax.set_xlim([-1, gs + 1])
    ax.set_ylim([-1, gs + 1])
    ax.grid(True)

    # Used to return the plot as an image rray
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


if __name__ == '__main__':
    main()
    print('Done.')