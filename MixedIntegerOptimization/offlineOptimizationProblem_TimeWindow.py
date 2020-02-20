# for technical file handling
import copy
import pickle
# for mathematical calculations and statistical distributions
from scipy.stats import truncnorm
from scipy.spatial.distance import cdist
import numpy as np
from gurobipy import *
# for graphics
import seaborn as sns
from matplotlib import pyplot as plt
from UtilsCode.createGif import create_gif
import time

sns.set()


def moveCar(carPos, targetPos):
    updatedCarPos = copy.copy(carPos)
    delta = targetPos - carPos
    if delta[0] != 0:
        updatedCarPos[0] += np.sign(delta[0])
    else:
        updatedCarPos[1] += np.sign(delta[1])
    return updatedCarPos


def calcReward(eventPos, carPos, closeReward, cancelPenalty, openedPenalty):
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
    nCars              = carPos.shape[0]
    nEvents            = eventPos.shape[0]
    distEventsToEvents = cdist(eventPos, eventPos, metric='cityblock')
    distCarsToEvents   = cdist(carPos, eventPos, metric='cityblock')

    rewardCarsToEvents   = -distCarsToEvents   + np.ones(shape=(nCars, nEvents))*closeReward
    rewardEventsToEvents = -distEventsToEvents + np.ones(shape=(nEvents, nEvents))*closeReward
    timeEventsToEvents   =  distEventsToEvents
    timeCarsToEvents     =  distCarsToEvents
    return rewardCarsToEvents, rewardEventsToEvents, timeCarsToEvents, timeEventsToEvents


def runMaxFlowOpt(tStart, carPos, eventPos , eventOpenTime, eventCloseTime,
                  closeReward, cancelPenalty, openedPenalty, outputFlag = 1):
    nEvents = eventPos.shape[0]
    nCars   = carPos.shape[0]
    rewardCarsToEvents, rewardEventsToEvents, timeCarsToEvents, timeEventsToEvents = calcReward(eventPos, carPos,
                                                                                                closeReward,
                                                                                                cancelPenalty,
                                                                                                openedPenalty)
    # Create optimization model
    m = Model('OfflineOpt')
    # Create variables
    x = m.addVars(nEvents, nEvents, name='cEventsToEvents', vtype=GRB.BINARY)
    y = m.addVars(nCars, nEvents, name='cCarsToEvents', vtype=GRB.BINARY)
    p = m.addVars(nEvents, name='isPickedUp', vtype=GRB.BINARY)
    t = m.addVars(nEvents, name='pickUpTime')

    # add constraint for events - maximum one event is picked up after each event
    # if p=0 then no events are picked up after,
    # if p=1 then one event is picked up after
    for i in range(nEvents):
        m.addConstr(sum(x[i, j] for j in range(nEvents)) <= p[i])

    # add constraint for cars - each car picks up maximum one event when starting
    # y_{k,c} is either 0 or 1 for all events
    for i in range(nCars):
        m.addConstr(sum(y[i, j] for j in range(nEvents)) <= 1)

    # add constraint for pickup - if event is picked up then it was either picked up first or after other event
    # p=0 - both sums need to add up to 0, no one picked this event up
    # p=1 - event is picked up first by car or after one other event
    for i in range(nEvents):
        sum1 = sum([y[c, i] for c in range(nCars)])
        sum2 = sum([x[e, i] for e in range(nEvents)])
        m.addConstr(p[i] == sum1+sum2)
    # add time constraint that the events are picked up at the time they are opened
    for i in range(nEvents):
        m.addConstr(t[i] >= eventOpenTime[i])
        m.addConstr(t[i] <= eventCloseTime[i])
    # add time constraint : time to pick up event is smaller or equal to the time event accurs
    for i in range(nEvents):
        for j in range(nEvents):
            m.addConstr(t[j]-t[i] >= (eventOpenTime[j]-eventCloseTime[i]) +
                                     (timeEventsToEvents[i, j] -
                                     (eventOpenTime[j]-eventCloseTime[i]))*x[i, j])
        for j in range(nCars):
            m.addConstr(t[i] >= eventOpenTime[i]+(tStart+timeCarsToEvents[j, i]-eventOpenTime[i])*y[j, i])

    # event can not be picked up after it is picked up (X_{c,c} = 0 by definition)
    for i in range(nEvents):
        m.addConstr(x[i, i] == 0)  # an event cant pick itself up

    rEvents       = 0  # reward for events that are closed after an event
    rCars         = 0  # reward for events that are closed after car
    pEvents       = 0  # penalty for events that are canceled
    pEventsOpened = 0  # penatly for time that events waited
    for i in range(nEvents):
        for j in range(nEvents):
            rEvents += rewardEventsToEvents[i, j]*x[i, j]
            pEventsOpened -= openedPenalty*(t[j]-eventOpenTime[j])*x[i, j]

        for j in range(nCars):
            rCars += rewardCarsToEvents[j, i]*y[j, i]
            pEventsOpened -= openedPenalty*(t[i] - eventOpenTime[i])*y[j, i]
        pEvents -= cancelPenalty*(1-p[i])

    obj = rEvents + rCars + pEvents + pEventsOpened
    m.setObjective(obj, GRB.MAXIMIZE)
    m.setParam('OutputFlag', outputFlag)
    m.setParam('LogFile', "")
    m.optimize()
    return m, obj


def poissonRandomEvents(startTime, endSimTime, lam):
    """
    creates time line assuming poisson distribution
    :param startTime: start time wanted for timeline
    :param endSimTime: end time wanted for timeline
    :param lam: lambda of poisson distribution (how often will an event appear?)
    :return: list of times for events
    """
    lengthTimeLine = int(endSimTime - startTime)
    numEventsPerTime = np.random.poisson(lam=lam, size=lengthTimeLine)
    eventTime = []
    for i in range(lengthTimeLine):
        nTime = numEventsPerTime[i]
        if nTime > 0:
            for num in range(nTime):
                eventTime.append(i+startTime)
    return np.array(eventTime)


def createEventsDistribution(gridSize, startTime, endTime, lam, eventTimeWindow):
    locX        = gridSize / 3
    scaleX      = gridSize / 3
    locY        = gridSize / 3
    scaleY      = gridSize / 3
    # randomize event times
    eventTimes  = poissonRandomEvents(startTime, endTime, lam)
    eventPosX   = truncnorm.rvs((0 - locX) / scaleX, (gridSize - locX) / scaleX, loc=locX, scale=scaleX,
                              size=len(eventTimes)).astype(np.int64)
    eventPosY   = truncnorm.rvs((0 - locY) / scaleY, (gridSize - locY) / scaleY, loc=locY, scale=scaleY,
                              size=len(eventTimes)).astype(np.int64)

    eventsPos           = np.column_stack([eventPosX, eventPosY])
    eventsTimeWindow    = np.column_stack([eventTimes, eventTimes+eventTimeWindow])
    return eventsPos,eventsTimeWindow


def plotResults(m, carsPos, eventsPos, eventsOpenTime, eventsCloseTime, plotFigures, fileLoc, fileName, gs):
    plot_gif = False
    if plot_gif:
        if not os.path.isdir(fileLoc + fileName):
            os.mkdir(fileLoc + fileName)
    nCars    = carsPos.shape[0]
    nEvents  = eventsPos.shape[0]
    paramKey = [v.varName.split('[')[0] for v in m.getVars()]
    param = {k: [] for k in paramKey}
    for v in m.getVars():
        param[v.varName.split('[')[0]].append(v.x)
    param['cEventsToEvents'] = np.array(param['cEventsToEvents']).reshape(nEvents, nEvents)
    param['cCarsToEvents']   = np.array(param['cCarsToEvents']).reshape(nCars, nEvents)
    param['isPickedUp']      = np.array(param['isPickedUp']).reshape(nEvents, 1)
    param['pickUpTime']      = np.array(param['pickUpTime']).reshape(nEvents, 1)
    eIds     = np.array(range(nEvents))
    pathCars = [[carsPos[i, :]] for i in range(nCars)]
    targetPos = []
    eventsPerCar = [[] for i in range(nCars)]
    for i in range(nCars):
        ePickedUp = eIds[param['cCarsToEvents'][i, :] == 1]
        if ePickedUp.size > 0:
            pathCars[i].append(eventsPos[ePickedUp, :][0])
            targetPos.append(eventsPos[ePickedUp, :][0])
            eventsPerCar[i].append(ePickedUp[0])
            carNotFinished = True
            while carNotFinished:
                ePickedUp = eIds[param['cEventsToEvents'][ePickedUp[0], :] == 1]
                if ePickedUp.size > 0:
                    pathCars[i].append(eventsPos[ePickedUp, :][0])
                    eventsPerCar[i].append(ePickedUp[0])
                else:
                    carNotFinished = False
        else:
            pathCars[i].append(carsPos[i, :])
            targetPos.append(carsPos[i, :])
        pathCars[i] = np.array(pathCars[i])
    carFullPath = [[carsPos[i, :]] for i in range(nCars)]
    currentPos  = copy.deepcopy(carsPos)
    maxTime = np.max(eventsCloseTime).astype(int)

    for c in range(nCars):
        k = 0
        for t in range(int(maxTime)):
            if cdist(currentPos[c, :].reshape(1, 2), targetPos[c].reshape(1, 2), metric='cityblock') > 0:
                # move car towards target
                currentPos[c, :] = moveCar(currentPos[c, :], targetPos[c])
            else:
                if len(eventsPerCar[c]) > 0:
                    # car has reached current target, check if target can be picked up
                    if t < param['pickUpTime'][eventsPerCar[c][k]]:
                        # target has not opened yet, needs to wait for target to open
                        targetPos[c] = copy.deepcopy(currentPos[c, :])
                    else:
                        if len(eventsPerCar[c])-1 > k:
                            # target is opened, advance car to next target
                            k += 1
                            targetPos[c] = copy.deepcopy(eventsPos[eventsPerCar[c][k], :])
                        else:
                            # car reached target and there are no more targets in cars list
                            targetPos[c] = copy.deepcopy(currentPos[c, :])
                        currentPos[c, :] = moveCar(currentPos[c, :], targetPos[c])
            carFullPath[c].append(copy.deepcopy(currentPos[c, :]))

    imageList = []
    if plotFigures:
        fig, ax = plt.subplots()
        ax.set_title('total results')
        ax.scatter([], [], c='y', marker = '*', label='Opened')
        ax.scatter([], [], c='k', marker = '*',label = 'Created')
        ax.scatter([], [], c='r', marker = '*', label='Canceled')
        ax.scatter([], [], c='g', marker = '*', label='Closed')

    numClosedVec = np.zeros(maxTime)
    numCanceledVec = np.zeros(maxTime)
    numOpenedVec = np.zeros(maxTime)
    numTotalEventsVec = np.zeros(maxTime)
    tempEventsVec = np.zeros(maxTime)
    for t in range(int(maxTime)):
        numEventsOpened   = 0
        numEventsClosed   = 0
        numEventsCanceled = 0
        for e in range(nEvents):
            if param['pickUpTime'][e]<=t and param['isPickedUp'][e]:
                numEventsClosed += 1
            if eventsCloseTime[e]<t and not param['isPickedUp'][e]:
                numEventsCanceled += 1
            if eventsOpenTime[e]<=t and eventsCloseTime[e]>=t:
                if not param['isPickedUp'][e]:
                    numEventsOpened += 1
                elif param['isPickedUp'][e] and param['pickUpTime'][e]>t:
                    numEventsOpened += 1
        numTotalEvents = numEventsClosed + numEventsCanceled + numEventsOpened
        numTotalEventsVec[t] = numTotalEvents
        numClosedVec[t] = numEventsClosed
        numOpenedVec[t] = numEventsOpened
        numCanceledVec[t] = numEventsCanceled
        if plot_gif:
            currentCarsPos = np.array([c[t] for c in carFullPath])
            plotForGif(currentCarsPos, eventsPos, param['pickUpTime'], eventsOpenTime, eventsCloseTime, param['isPickedUp'], fileName, t, gs)
    timeVec = np.array(range(int(maxTime)))
    dataOut = {'closedEvents'       : numClosedVec,
               'canceledEvents'     : numCanceledVec,
               'openedEvents'       : numOpenedVec,
               'allEvents'          : numTotalEventsVec,
               'time'               : timeVec}
    if plotFigures:
        plt.plot(timeVec, numClosedVec,      c='g', marker='*')
        plt.plot(timeVec, numCanceledVec,    c='r', marker='*')
        plt.plot(timeVec, numOpenedVec,      c='y', marker='*')
        plt.plot(timeVec, numTotalEventsVec, c='k', marker='*')
        ax.grid(True)
        plt.legend()
        plt.show()

    if plot_gif:
        listNames = [fileName + '_' + str(t) + '.png' for t in range(int(maxTime))]
        create_gif(fileLoc + fileName + '/', listNames, 1, fileName)
    return dataOut, param, carFullPath


def plotForGif(carPos, eventPos, eventPickUpTime, eventOpenTime, eventCloseTime, isEventPicked, fileName, t, gs):
    """

    :param carPos: position of cars at time t (nc,2)
    :param eventPos: position of events (ne,2)
    :param eventPickUpTime: time that event was picked up
    :param eventOpenTime: time event started being opened
    :param eventCloseTime: time event closed if not picked up
    :param isEventPicked: True/False if event is picked up or not
    :param gs: grid size
    :param t: current time to plot
    :return: graph to add to list of graphs
    """
    fig, ax = plt.subplots()
    ax.set_title('time: {0}'.format(t))
    for c in range(carPos.shape[0]):
        ax.scatter(carPos[c,0], carPos[c,1], c='k', alpha=0.5)
    ax.scatter([], [], c='b', marker='*', label='Opened not commited')
    ax.scatter([], [], c='b', label='Opened commited')
    ax.scatter([], [], c='r', label='Canceled')
    ax.scatter([], [], c='g', label='Closed')
    for i in range(eventPos.shape[0]):

        if t >= eventPickUpTime[i] and isEventPicked[i]:
            ax.scatter(eventPos[i, 0], eventPos[i, 1], c='g', alpha=0.2)
        elif (eventOpenTime[i] <= t) and (eventCloseTime[i] > t):
            ax.scatter(eventPos[i, 0], eventPos[i, 1], c='b', alpha=0.7)
        elif t>eventCloseTime[i]:
            ax.scatter(eventPos[i, 0], eventPos[i, 1], c='r', alpha=0.2)
        else:
            ax.scatter(eventPos[i, 0], eventPos[i, 1], c='y', alpha=0.2)
    ax.set_xlim([-1, gs[0] + 1])
    ax.set_ylim([-1, gs[1] + 1])
    ax.grid(True)
    plt.legend()
    plt.savefig(fileName + '_' + str(t)+'.png')
    plt.close()
    return


def main():
    # carPos              = np.array([0, 0]).reshape(nCars,2)
    # eventPos            = np.array([[5, 0], [5, 5]]).reshape((nEvents,2))
    # eventTime           = np.array([5, 8])
    closeReward = 50
    cancelPenalty = 100
    openedPenalty = 5
    sim_seed      = 1
    np.random.seed(sim_seed)
    gridSize            = [10, 10]
    nCars               = 2
    tStart              = 0
    deltaOpenTime       = 12
    lengthSim           = 15
    lam                 = 1.7  # 2/3

    plotFigures = True
    fileLoc = '/Users/chanaross/dev/Thesis/MixedIntegerOptimization/'
    fileName = 'optimizationResults_small'

    carPos = np.reshape(np.random.randint(0, gridSize[0], 2 * nCars), (nCars, 2))

    eventPos, eventTimes = createEventsDistribution(gridSize[0], 0, lengthSim, lam, deltaOpenTime)
    eventStartTime = eventTimes[:, 0]
    eventEndTime = eventTimes[:, 1]
    s_time = time.time()
    m, obj = runMaxFlowOpt(tStart, carPos, eventPos,
                           eventStartTime, eventEndTime,
                           closeReward, cancelPenalty, openedPenalty)
    e_time = time.time()
    print("sim run time : "+str(e_time - s_time))
    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % obj.getValue())
    dataOut, param, cars_paths = plotResults(m, carPos, eventPos, eventStartTime, eventEndTime, plotFigures, fileLoc, fileName, gridSize)
    with open(fileLoc + fileName + '.p', 'wb') as out:
        pickle.dump({'sol': param,
                     'cars_paths': cars_paths,
                     'sol_post_process' : dataOut,
                     'n_cars': nCars,
                     'sim_time': lengthSim,
                     'gs': gridSize,
                     'lam':lam,
                     'del_open_time': deltaOpenTime,
                     'seed': sim_seed,
                     'event_pos': eventPos,
                     'event_start_time': eventStartTime,
                     'event_end_time': eventEndTime
                     }, out)

if __name__ == '__main__':
    main()
    print('Done.')
