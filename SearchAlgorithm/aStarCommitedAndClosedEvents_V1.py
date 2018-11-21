from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import comb
import time
import copy
import itertools
import imageio
import heapq
import pickle
sns.set()





__author__ = "Chana Ross, Yoel Ross"
__copyright__ = "Copyright 2018"
__credits__ = ["Yoel Ross", "Tamir Hazan", "Erez Karpas"]
__version__ = "1.0"
__maintainer__ = "Chana Ross"
__email__ = "schanaby@campus.technion.ac.il"
__status__ = "Thesis"


# Class definitions
class SearchState:
    def __init__(self, carPos, committedCars, eventPos, committedEventsIndex, committedEvents, eventStartTimes ,eventCloseTimes, eventReward, eventPenalty, eventsCanceled, eventsAnswered, heuristicVal, costVal, parent,hWeight):
        self.carPos                 = carPos
        self.committedCars          = committedCars
        self.committedEventsIndex   = committedEventsIndex
        self.hWeight                = hWeight
        self.eventPos               = eventPos
        self.committedEvents        = committedEvents
        self.eventStartTimes        = eventStartTimes
        self.eventCloseTimes        = eventCloseTimes
        self.eventReward            = eventReward
        self.eventPenalty           = eventPenalty
        self.eventsCanceled         = eventsCanceled   # true if the event is opened, false if event closed or canceled. starts as true
        self.eventsAnswered         = eventsAnswered   # true if event is answered  , false if event is canceled. starts as false
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
        carEq       = np.array_equal(self.carPos, other.carPos)
        carComEq    = np.array_equal(self.committedCars,other.committedCars)
        eventComEq  = np.array_equal(self.committedEvents,other.committedEvents)
        eveEq       = np.array_equal(self.eventPos,other.eventPos)
        etmEq       = np.array_equal(self.eventTimes,other.eventTimes)
        comEveIndex = np.array_equal(self.committedEventsIndex,other.committedEventsIndex)
        etcEq       = np.array_equal(self.eventCloseTimes,other.eventCloseTimes)
        sttEq       = np.array_equal(self.eventsCanceled, other.eventsCanceled)
        astEq       = np.array_equal(self.eventsAnswered,other.eventsAnswered)
        timEq       = self.time == other.time
        return carEq and eveEq and etcEq and etmEq and eventComEq and carComEq and sttEq and comEveIndex and astEq and timEq

    def __repr__(self):
        return "time: {0}, cost: {1}, heuristic: {2}, root: {3}, goal: {4}\n".format(self.time,
                                                                                     self.gval,
                                                                                     self.hval,
                                                                                     self.root,
                                                                                     self.goalCheck())

    def __hash__(self):
        carPosVec   = np.reshape(self.carPos, self.carPos.size)
        evePosVec   = np.reshape(self.eventPos, self.eventPos.size)
        eveComVec   = np.reshape(self.committedEvents, self.committedEvents.size)
        carComVec   = np.reshape(self.committedCars,self.committedCars.size)
        comEveIndex = np.reshape(self.committedEventsIndex,self.committedEventsIndex.size)
        eveSttVec   = self.eventsCanceled.astype(np.int32)
        eveAnStVec  = self.eventsAnswered.astype(np.int32)
        stateTime   = np.reshape(np.array(self.time), 1)
        hv = np.hstack([carPosVec , evePosVec , comEveIndex , eveSttVec ,eveAnStVec , stateTime, carComVec, eveComVec])
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




def updateStatusVectors(distanceMatrix,canceledEvents,answeredEvents,eventsCloseTime,eventsStartTime,epsilon = 0.01):
    eventsOpened   = np.logical_and(eventsStartTime<=0, eventsCloseTime>=0)
    # convert distance matrix to boolean of approx zero (picked up events)
    step1          = np.sum((distanceMatrix <= epsilon), axis=0) >= 1
    # condition on event being open
    step2          = np.logical_and(step1, eventsOpened)
    updatedEventsAnswered = np.copy(answeredEvents)
    # new possible events answered status
    updatedEventsAnswered[step2] = 1
    updatedEventsAnswered.astype(np.bool_)
    numAnsweredEvents = np.sum(step2)
    # find canceled events and add to canceled vector
    step1Canceled = np.logical_and(np.logical_not(answeredEvents),eventsCloseTime<0)
    step2Canceled = np.logical_and(step1Canceled,np.logical_not(canceledEvents))
    updatedEventsCanceled = np.copy(canceledEvents)
    updatedEventsCanceled[step2Canceled] = 1
    updatedEventsCanceled.astype(np.bool_)
    numCanceledEvents = np.sum(step2Canceled)
    updatedEventsOpened = np.logical_and(np.logical_not(step1,eventsOpened))
    numOpenedEvents = np.sum(updatedEventsOpened)
    return updatedEventsAnswered,updatedEventsCanceled,numAnsweredEvents,numCanceledEvents,numOpenedEvents

def updateCommittedStatus(committedEvents,committedCars,committedEventIndex,canceledEvents,answeredEvents):
    eventClosed = np.logical_or(canceledEvents,answeredEvents)

    updatedCommittedCars = np.copy(committedCars)
    updatedCommittedCars[updatedCommittedCars>0] = np.logical_not(eventClosed[committedEventIndex>=0])

    updatedCommittedEvents = np.copy(committedEvents)
    updatedCommittedEvents[eventClosed] = False

    updatedCommittedEventIndex = np.copy(committedEventIndex)
    updatedCommittedEventIndex[np.logical_not(updatedCommittedCars)] = -1
    return updatedCommittedCars,updatedCommittedEvents,updatedCommittedEventIndex


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





