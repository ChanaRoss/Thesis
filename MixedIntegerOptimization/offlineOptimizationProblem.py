import numpy as np
from matplotlib import pyplot as plt
from gurobipy import *
# for mathematical calculations and statistical distributions
from scipy.stats import truncnorm
from scipy.spatial.distance import cdist
from scipy.special import comb
import math
import copy
import itertools

def createCarFullPath(pathCars):
    fullPath = []
    for c in range(len(pathCars)):
        i = 0
        carPos = copy.copy(pathCars[c][0])
        fullPath.append([carPos])
        while i<pathCars[c].shape[0]-1:
            tempPos = copy.copy(pathCars[c][i])
            targetPos = copy.copy(pathCars[c][i+1])
            while not np.array_equal(tempPos,targetPos):
                delta = targetPos - tempPos
                if delta[0] != 0:
                    tempPos[0] += np.sign(delta[0])
                else:
                    tempPos[1] += np.sign(delta[1])
                fullPath[c].append(tempPos)
            i += 1
    return fullPath


def calcReward(eventPos,carPos,closeReward,cancelPenalty,openedPenalty):
    """
    this function calculates the reward that will be achieved assuming event is picked up
    :param eventPos: position of events
    :param carPos: position of cars
    :param closeReward: reward if event is closed
    :param cancelPenalty: penalty if event is canceled (for now assuming events are not canceled)
    :param openedPenalty: penalty for time events are waiting (for now assuming events dont wait since they are picked up as spesific time)
    :return: rewardCarsToEvents -   R_{cars,events},
             rewardEventsToEvents - R_{events,events}
    """
    nCars   = carPos.shape[0]
    nEvents = eventPos.shape[0]
    distEventsToEvents = cdist(eventPos,eventPos,metric = 'cityblock')
    distCarsToEvents   = cdist(carPos,eventPos,metric='cityblock')

    rewardCarsToEvents   = -distCarsToEvents + np.ones(shape = (nCars,nEvents))*closeReward
    rewardEventsToEvents = -distEventsToEvents + np.ones(shape = (nEvents,nEvents))*closeReward
    timeEventsToEvents    = distEventsToEvents
    timeCarsToEvents     = distCarsToEvents
    return rewardCarsToEvents,rewardEventsToEvents,timeCarsToEvents,timeEventsToEvents



def runMaxFlowOpt(tStart,carPos,eventPos,eventTime,closeReward,cancelPenalty,openedPenalty):
    nEvents = eventPos.shape[0]
    nCars   = carPos.shape[0]
    rewardCarsToEvents, rewardEventsToEvents, timeCarsToEvents, timeEventsToEvents = calcReward(eventPos,carPos,closeReward,cancelPenalty,openedPenalty)
    # Create optimization model
    m = Model('OfflineOpt')
    # Create variables
    x = m.addVars(nEvents,nEvents,name = 'cEventsToEvents',vtype=GRB.BINARY)
    y = m.addVars(nCars,nEvents,name = 'cCarsToEvents',vtype=GRB.BINARY)
    p = m.addVars(nEvents,name = 'isPickedUp',vtype=GRB.BINARY)
    t = m.addVars(nEvents,name = 'pickUpTime')

    # add constraint for events - maximum one event is picked up after each event
    # if p=0 then no events are picked up after,
    # if p=1 then one event is picked up after
    for i in range(nEvents):
        m.addConstr(sum(x[i,j] for j in range(nEvents))<=p[i])

    # add constraint for cars - each car picks up maximum one event when starting
    # y_{k,c} is either 0 or 1 for all events
    for i in range(nCars):
        m.addConstr(sum(y[i,j] for j in range(nEvents))<=1)
    # add constraint for pickup - if event is picked up then it was either picked up first or after other event
    # p=0 - both sums need to add up to 0, no one picked this event up
    # p=1 - event is picked up first by car or after one other event
    for i in range(nEvents):
        sum1 = sum([y[c,i] for c in range(nCars)])
        sum2 = sum([x[e,i] for e in range(nEvents)])
        m.addConstr(p[i] == sum1+sum2)
    # add time constraint that the events are picked up at the time they are opened
    for i in range(nEvents):
        m.addConstr(t[i] == eventTime[i])
    # add time constraint : time to pick up event is smaller or equal to the time event accurs
    for i in range(nEvents):
        for j in range(nEvents):
            m.addConstr((timeEventsToEvents[i,j]-(t[j]-t[i]))*x[i,j]<=0)
        for j in range(nCars):
            m.addConstr((tStart-t[i]+timeCarsToEvents[j,i])*y[j,i]<=0)
    # event can not be picked up after it is picked up (X_{c,c} = 0 by definition)
    for i in range(nEvents):
        m.addConstr(x[i,i] == 0) # an event cant pick itself up



    rEvents = 0
    rCars   = 0
    pEvents = 0
    for i in range(nEvents):
        for j in range(nEvents):
            rEvents += rewardEventsToEvents[i,j]*x[i,j]
        for j in range(nCars):
            rCars += rewardCarsToEvents[j,i]*y[j,i]
        pEvents -= cancelPenalty*(1-p[i])

    obj = rEvents + rCars + pEvents
    m.setObjective(obj, GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    m.setParam('LogFile', "")
    m.optimize()

    # for v in m.getVars():
    #     print('%s %g' % (v.varName, v.x))
    #
    # print('Obj: %g' % obj.getValue())

    return m, obj


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

def plotResults(m,carsPos,eventsPos):
    nCars = carsPos.shape[0]
    nEvents = eventsPos.shape[0]
    paramKey = [v.varName.split('[')[0] for v in m.getVars()]
    param = {k:[] for k in paramKey}
    for v in m.getVars():
        param[v.varName.split('[')[0]].append(v.x)
    param['cEventsToEvents'] = np.array(param['cEventsToEvents']).reshape(nEvents,nEvents)
    param['cCarsToEvents']   = np.array(param['cCarsToEvents']).reshape(nCars,nEvents)
    param['isPickedUp']      = np.array(param['isPickedUp']).reshape(nEvents,1)
    param['pickUpTime']      = np.array(param['pickUpTime']).reshape(nEvents, 1)
    eIds     = np.array(range(nEvents))
    pathCars = [[carsPos[i,:]] for i in range(nCars)]
    for i in range(nCars):
        ePickedUp = eIds[param['cCarsToEvents'][i,:] == 1]
        if ePickedUp.size>0:
            pathCars[i].append(eventsPos[ePickedUp,:][0])
        carNotFinished = True
        while carNotFinished:
            ePickedUp = eIds[param['cEventsToEvents'][ePickedUp[0],:]== 1]
            if ePickedUp.size>0:
                pathCars[i].append(eventsPos[ePickedUp,:][0])
            else:
                carNotFinished = False
        pathCars[i] = np.array(pathCars[i])
    carFullPath = createCarFullPath(pathCars)
    maxTime = np.max(param['pickUpTime']).astype(int)
    plt.figure()
    tempCarPos = carsPos
    for e in range(nEvents):
        plt.plot(eventsPos[e,0],eventsPos[e,1],marker = 'o',color = 'k',label = 'EventId:'+str(e))
    for i in range(maxTime):
        for c in range(nCars):
            plt.plot(tempCarPos[c,0],tempCarPos[c,1],marker = '*',color = 'm',label = 'carId:'+str(c))
    print('hi')
    return

def main():
    # carPos              = np.array([0, 0]).reshape(nCars,2)
    # eventPos            = np.array([[5, 0], [5, 5]]).reshape((nEvents,2))
    # eventTime           = np.array([5, 8])
    closeReward = 50
    cancelPenalty = 100
    openedPenalty = 1

    np.random.seed(1)
    gridSize            = 20
    nCars = 5
    tStart              = 0
    deltaOpenTime       = 0
    lengthSim           = 20
    lam                 = 2/3


    carPos = np.reshape(np.random.randint(0, gridSize, 2 * nCars), (nCars, 2))

    eventPos, eventTimes = createEventsDistribution(gridSize, 0, lengthSim, lam, deltaOpenTime)
    eventStartTime = eventTimes[:, 0]
    eventEndTime = eventTimes[:, 1]

    m, obj = runMaxFlowOpt(tStart,carPos,eventPos,eventStartTime,closeReward,cancelPenalty,openedPenalty)

    plotResults(m,carPos,eventPos)

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % obj.getValue())



if __name__ == '__main__':
    main()
    print('Done.')
