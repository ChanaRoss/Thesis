import numpy as np
from matplotlib import pyplot as plt
import sys
import pickle
import math
import copy
sys.path.insert(0, '/home/chanaby/Documents/Thesis/aima-python')
from search import Problem,astar_search
from UtilsShortestPath import *

class SolveCars2EventsAstar(Problem):
    """ The problem of moving the Hybrid Wumpus Agent from one place to other """

    def __init__(self, initial, gridSize):
        """ Define goal state and initialize a problem """
        self.gridSize = gridSize
        self.numEvents = len([s for s in initial if isinstance(s,tuple) if 'event' in s])
        self.numCars = len([s for s in initial if isinstance(s,tuple) if 'car' in s])
        self.epsilon = 0.01
        Problem.__init__(self, initial)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result will be a list"""
        actionFlag = state[-1]
        actionOptions = [[1, 0], [-1, 0], [0, 1], [0, -1]]  # all the options for the car movements
        possibleActions = []
        if actionFlag < self.numCars:
            for movement in actionOptions:
                newState = copy.deepcopy(state)  # copy the old state in order to update the location of the car
                car2Move = newState[actionFlag]  # this is the car that will be moved at this spesific state
                carLoc = car2Move[1]
                checkOutOfWidth = np.array(carLoc)[0] + np.array(movement)[0]
                checkOutOfHeight = np.array(carLoc)[1] + np.array(movement)[1]
                if 0<=checkOutOfWidth<=self.gridSize and 0<=checkOutOfHeight<=self.gridSize:
                    car2MoveList = list(car2Move)
                    car2MoveList[1] = tuple(np.array(carLoc) + np.array(movement))
                    car2Move = tuple(car2MoveList)
                    newStateList = list(newState)
                    newStateList[actionFlag] = car2Move  # update location of the car
                    newState = tuple(newStateList)
                    for i, s in enumerate(newState):
                        if isinstance(s,tuple) and 'event' in s and self.manhattenDistance(car2Move[1], s[1]) < self.epsilon and s[2]<= 0:
                            # print('event #' + str(s[1]) +' reached')
                            newStateList = list(newState)
                            sList = list(s)
                            sList[3] = True  # this event is closed since a car has reached its location
                            newStateList[i] = tuple(sList)
                            newState = tuple(newStateList)
                    newStateList = list(newState)
                    newStateList[-1] += 1
                    newState = tuple(newStateList)
                    possibleActions.append(newState)  # add the new state to the list of possible actions
        else:
            stateList = list(state)
            stateList[-1] = 0
            for i, s in enumerate(stateList):
                if isinstance(s,tuple) and 'event' in s:
                    sList = list(s)
                    sList[2] -= 1
                    s = tuple(sList)
                    stateList[i] = s  # shorten the time by 1
            state = tuple(stateList)
            possibleActions.append(state)

        return possibleActions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """
        # we define our action as a state already and therefore the new state is the action itself
        return action

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        flagAction = state2[-1]
        if flagAction<self.numCars:
            # the cost should be the number of steps that each car has moved
            costOfAction = self.manhattenDistance(state2[flagAction][1],state1[flagAction][1])
        else:
            costOfAction = 0
            for i,s in enumerate(state2):
                if isinstance(s,tuple) and 'event' in s:
                    if s[2]<0 and s[3] == False:
                        costOfAction += 1 # add one to cost if event is opnened and no car has reached it yet
        return c + costOfAction

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """
        flagIsGoal = False
        numClosedEvents = 0
        for s in state:
            if isinstance(s,tuple) and 'event' in s and s[3]:
                numClosedEvents +=1
        if numClosedEvents == self.numEvents:
            flagIsGoal = True
        return flagIsGoal

    def h(self, node):
        """ Return the heuristic value for a given state."""
        # the heuristic is the minimum between: {(distance from closest car to event)-(time until event openes),0}
        # therefore if the car will reach the event earlier the heuristic is 0, otherwise its the difference between the distance and time when event opens
        totalHeuristic = 0
        cars = [s for s in node.state if isinstance(s,tuple) and 'car' in s]
        for s in node.state:
            if isinstance(s,tuple) and 'event' in s:
                minDistanceCar2Event = math.inf
                for car in cars:
                    distance2event = self.manhattenDistance(car[1],s[1])
                    if distance2event<minDistanceCar2Event:
                        minDistanceCar2Event = distance2event
                timeToOpenEvent = s[2]
                heuristicForEvent = np.min(minDistanceCar2Event-timeToOpenEvent,0)
                totalHeuristic += heuristicForEvent
        return totalHeuristic

    def manhattenDistance(self,loc1,loc2):
        return abs(loc2[0]-loc1[0]) + abs(loc2[1]-loc1[1])


def createInitialState(carsLoc,eventsTime,eventsLoc,flagActions):
    initState = []
    for c in carsLoc:
        temp = ('car', c)
        initState.append(temp)

    for t, e in zip(eventsTime, eventsLoc):
        temp = ('event', e, t, False)
        initState.append(temp)

    initState.append(flagActions)
    return tuple(initState)

def main():
    carsLoc    = ((0,0),(2,1))
    eventsLoc  = ((1,1),(2,2,),(0,2))
    gridSize = 5
    eventsTime = (1,4,4)
    flagActions = 0
    initial = createInitialState(carsLoc,eventsTime,eventsLoc,flagActions)
    solver = SolveCars2EventsAstar(initial = initial,gridSize=gridSize)

    astarSolution = astar_search(solver, h=None)

    CarShortestPath = PrintShortestPath(astarSolution)
    # dump logs
    with open('aStarResult.p', 'wb') as out:
        pickle.dump({'shortestPath':CarShortestPath,'aimaSolution':astarSolution}, out)


if __name__ == '__main__':
    main()
    print('im done!')