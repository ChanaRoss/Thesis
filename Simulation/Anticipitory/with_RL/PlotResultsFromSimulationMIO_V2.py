import imageio
# import my file in order to load state class from pickle
from Simulation.Anticipitory.with_RL.simAnticipatoryWithMIO_poisson_RL import *
from UtilsCode.createGif import create_gif
from MixedIntegerOptimization.offlineOptimizationProblem_TimeWindow import runMaxFlowOpt, plotResults

sns.set()
#  load logs
pickleNames = []

# test results on new code
# pickleNames.append('SimAnticipatoryMio_RandomChoice_5lpred_5delOpen_48startTime_20gridX_20gridY_31numEvents_20nStochastic_4numCars_Bm_poisson_TimeWindow')
# pickleNames.append('SimGreedy_5lpred_5delOpen_48startTime_20gridX_20gridY_31numEvents_20nStochastic_4numCars_Bm_poisson_TimeWindow')
# pickleNames.append('SimOptimization_MaxFlow_5lpred_5delOpen_48startTime_20gridX_20gridY_31numEvents_20nStochastic_4numCars_Bm_poisson_TimeWindow')
# pickleNames.append('SimOptimization_TimeWindow_5lpred_5delOpen_48startTime_20gridX_20gridY_31numEvents_20nStochastic_4numCars_Bm_poisson_TimeWindow')


# comparison to RL -
pickleNames.append('SimAnticipatoryMio_RandomChoice_5lpred_5delOpen_48startTime_20gridX_20gridY_9numEvents_20nStochastic_2numCars_Bm_poisson_TimeWindow')
pickleNames.append('SimGreedy_5lpred_5delOpen_48startTime_20gridX_20gridY_9numEvents_20nStochastic_2numCars_Bm_poisson_TimeWindow')
pickleNames.append('SimOptimization_MaxFlow_5lpred_5delOpen_48startTime_20gridX_20gridY_9numEvents_20nStochastic_2numCars_Bm_poisson_TimeWindow')
pickleNames.append('SimOptimization_TimeWindow_5lpred_5delOpen_48startTime_20gridX_20gridY_9numEvents_20nStochastic_2numCars_Bm_poisson_TimeWindow')



def filterEvents(eventDict, currentTime,lg):
    filterdEventDict = {}
    for event in eventDict.values():
        currentStatusLog = [s for s in event['statusLog'] if float(s[0]) == currentTime][0]
        if currentStatusLog[1]:
            filterdEventDict[event['id']] = event
    return filterdEventDict

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
    return dx,dy

def manhattenDist(position1, position2):
    """
    calc manhatten distacne between two positions
    :param position1:
    :param position2:
    :return: distance (int)
    """
    dx,dy = manhattenPath(position1, position2)
    return float(abs(dx) + abs(dy))

def plotCurrentTimeGreedy(time,gridSize,lg):

    cars     = lg['cars']
    events   = lg['events']
    eventLog = lg['eventLog']
    fig, ax  = plt.subplots()
    # plot car positions
    for cid in cars:
        pos = [p['position'] for p in cars[cid] if int(p['time']) == time][0]
        ax.scatter(pos[0], pos[1], c='r', alpha=0.5, label='cars')
        ax.text(pos[0], pos[1] + 0.1, 'cid: {0}'.format(cid))
    # plot event positions
    filteredEvents = filterEvents(lg['events'], time,lg)
    for fev in filteredEvents.values():
        ePos = fev['position']
        ax.scatter(ePos[0], ePos[1], c='b', alpha=0.5, label='events')
        ax.text(ePos[0], ePos[1] + 0.1, 'eid: {0}'.format(fev['id']))
    ax.set_title('current time: {0}'.format(time))
    ax.grid(True)
    ax.legend()
    ax.set_ylim([-3, gridSize + 3])
    ax.set_xlim([-3, gridSize + 3])

    # Used to return the plot as an image rray
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def plotCarsHeatmap(gridSize,lg,simTime,pickleName):
    heatMat = np.zeros(shape=(gridSize[0], gridSize[1]))
    if 'SimAnticipatory' in pickleName or 'Greedy' in pickleName:
        carsPos = [c.path for c in lg['pathresults'][-1].cars.notCommited.values()]
        for carPos in carsPos:
            for pos in carPos:
                heatMat[int(pos[0]),int(pos[1])] +=1
            plt.figure()
            sns.heatmap(heatMat)
            plt.title('Heat map of car location - anticipatory')
    elif 'Hungarian' in pickleName:
        cars = lg['cars']
        for car in cars.values():
            carPosition = [c['position'] for c in car if c['time']<=simTime]
            for pos in carPosition:
                heatMat[int(pos[0]),int(pos[1])] +=1
            plt.figure()
            sns.heatmap(heatMat)
            plt.title('Heat map of car location - greedy')

def plotBasicStatisticsOfEvents(gridSize,lg,pickleName,simTime, fig, ax):
    plotAllEvents = False
    if 'Hungarian' in pickleName or 'Greedy' in pickleName:
        labelStr = 'Greedy Algorithm'
        lineStyle = '--'
        m_str="o"
    elif 'SimAnticipatory' in pickleName and 'Bm_easy' in pickleName:
        # labelStr = 'Anticipatory Algorithm based on Benchmark - seq'
        labelStr = 'Anticipatory Algorithm - Prior Knowledge'
        lineStyle = '-'
        plotAllEvents = True
        m_str="s"
    elif 'SimAnticipatory' in pickleName and 'NN' in pickleName:
        # labelStr = 'Anticipatory Algorithm based on NN'
        labelStr = 'Anticipatory Algorithm - NN'
        lineStyle = '-'
        plotAllEvents = True
        m_str="s"
    elif 'SimAnticipatory' in pickleName and 'Bm' in pickleName:
        # labelStr = 'Anticipatory Algorithm based on Benchmark'
        labelStr = 'Anticipatory Algorithm - Prior Knowledge'
        lineStyle = '-'
        plotAllEvents = True
        m_str="s"
    elif 'Optimization' in pickleName and 'TimeWindow' in pickleName:
        # labelStr = 'determinsitc MIO results with time window'
        labelStr = 'MIO - with time window'
        lineStyle = ':'
        m_str ="d"
    elif 'Optimization' in pickleName and 'MaxFlow' in pickleName:
        # labelStr = 'determinsitc MIO results - max flow'
        labelStr = 'MIO - max flow'
        lineStyle = ':'
        m_str='d'
    if 'SimAnticipatory' in pickleName or 'Greedy' in pickleName or 'Optimization' in pickleName:
        if plotAllEvents:
            ax.plot(lg['time'], lg['allEvents'], c='k', marker='*', markersize=8, linewidth=1.5, label='Num Total Events')
            plotAllEvents = False
        ax.plot(lg['time'], lg['closedEvents'], linestyle=lineStyle, linewidth=1.5, marker=m_str, label=labelStr)
        # plt.plot(lg['time'], lg['canceledEvents'], c='y', linestyle=lineStyle, label='canceled')
        ax.legend(loc='upper left', fontsize="medium")
        ax.grid(True)
        ax.set_xlabel('time', fontsize=20)
        ax.set_ylabel('num events', fontsize=20)
        ax.set_title('number of events over time', fontsize=20)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)

        # current over time
        plt.figure(3)
        plt.plot(lg['time'], lg['OpenedEvents'], label='current for :' + labelStr)
        plt.title('currently open over time')
        plt.xlabel('time')
        plt.ylabel('num events')
        plt.grid(True)
        plt.legend()
        if 'Optimization' not in pickleName:
            eventsPos = [c.position for c in lg['pathresults'][-1].events.notCommited.values()]
            eventsStartTime = [c.startTime for c in lg['pathresults'][-1].events.notCommited.values()]
            eventsEndTime = [c.endTime for c in lg['pathresults'][-1].events.notCommited.values()]
            eventsStatus = [c.status for c in lg['pathresults'][-1].events.notCommited.values()]
            plt.figure(4)
            for (i,ePos,eStatus) in zip(range(len(eventsPos)),eventsPos,eventsStatus):
                if eStatus == Status.CLOSED:
                    plt.scatter(ePos[0],ePos[1], color='g', label=str(i))
                    plt.text(ePos[0], ePos[1], i)
                else:
                    plt.scatter(ePos[0], ePos[1], color='r', label=str(i))
                    plt.text(ePos[0], ePos[1], i)
            plt.xlabel('grid X')
            plt.ylabel('grid Y')
            plt.title('event locations for: ' + labelStr)
            plt.xlim([-3, gridSize[0] + 3])
            plt.ylim([-3, gridSize[1] + 3])

    if 'Hungarian' in pickleName:
        eventLog    = lg['eventLog']
        carLog      = lg['cars']
        eventDict   = lg['events']
        timeLine    = np.unique([c['time'] for c in carLog[0] if c['time']<=simTime])
        lenEventLog = len([e for e in eventLog['time'] if e<=simTime])
        plotCount   = eventLog['count'][:lenEventLog]
        plotClosed  = eventLog['closed'][:lenEventLog]
        plotCurrent = eventLog['current'][:lenEventLog]
        plotingTimeline = timeLine
        plt.figure(2)
        plt.scatter(plotingTimeline, plotCount, c='r', label='Num Created events')
        plt.plot(plotingTimeline, plotClosed, label='Num Closed for :'+labelStr)
        plt.plot(plotingTimeline, eventLog['canceled'], c='y', label='canceled')
        plt.legend()
        plt.grid(True)
        plt.xlabel('time')
        plt.ylabel('num events')
        plt.title('number of events over time')

        # current over time
        plt.figure(3)
        plt.plot(plotingTimeline,plotCurrent , label='current for :' + labelStr)
        plt.title('currently open over time')
        plt.xlabel('time')
        plt.ylabel('num events')
        plt.grid(True)
        plt.legend()


        plt.figure(4)
        for event in eventDict.values():
            if event['closed']:
                plt.scatter(event['position'][0], event['position'][1], color='g', label=event['id'])
                plt.text(event['position'][0], event['position'][1], event['id'])
            else:
                plt.scatter(event['position'][0], event['position'][1], color='r', label=event['id'])
                plt.text(event['position'][0], event['position'][1], event['id'])
        plt.xlabel('grid X')
        plt.ylabel('grid Y')
        plt.title('event locations for: ' + labelStr)
        plt.xlim([-3, gridSize + 3])
        plt.ylim([-3, gridSize + 3])


def plotCurrentTimeAnticipatory(s, ne, nc, gs,fileName):
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
        ax.scatter(carTemp.position[0], carTemp.position[1], c='k', alpha=1, marker='s')
    ax.scatter([], [], c='y', label='Future Requests')
    ax.scatter([], [], c='b', label='Opened')
    # ax.scatter([], [], c='b', label='Opened commited')
    ax.scatter([], [], c='r', label='Canceled')
    ax.scatter([], [], c='g', label='Closed')
    for i in range(ne):
        eventTemp = s.events.getObject(i)
        if eventTemp.status == Status.OPENED_COMMITED:
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='b', alpha=0.8)
        elif eventTemp.status == Status.OPENED_NOT_COMMITED:
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='b', alpha=1)
        elif (eventTemp.status == Status.CLOSED):
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='g', alpha=0.6)
        elif (eventTemp.status == Status.CANCELED):
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='r', alpha=0.6)
        else:
            ax.scatter(eventTemp.position[0], eventTemp.position[1], c='y', alpha=0.7, marker='+')
    ax.set_xlim([-1, gs[0] + 1])
    ax.set_ylim([-1, gs[1] + 1])
    ax.grid(True)
    plt.legend()
    plt.savefig(fileName + '_' + str(s.time)+'.png')
    plt.close()
    return
    # # Used to return the plot as an image rray
    # fig.canvas.draw()  # draw the canvas, cache the renderer
    # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # return image

def optimizedSimulation(initialState,numFigure):
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

    m, obj = runMaxFlowOpt(0, carsPos, np.array(eventsPos), np.array(eventsStartTime), np.array(eventsEndTime), initialState.closeReward,
                           initialState.cancelPenalty, initialState.openedNotCommitedPenalty, 0)
    plotResults(m, carsPos, np.array(eventsPos), np.array(eventsStartTime), np.array(eventsEndTime), numFigure)


def main():
    imageList = []

    FlagCreateGif = 0
    fileLoc = '/Users/chanaross/dev/Thesis/Simulation/Anticipitory/with_RL/Results/'
    fig, ax = plt.subplots(1, 1)
    for pickleName in pickleNames:
        lg  = pickle.load(open(fileLoc + pickleName + '.p', 'rb'))
        simTime = 20
        if 'Hungarian' in pickleName:
            events   = lg['events']
            gridSize = lg['gs']
            simTime  = np.min([np.max([c['time'] for c in lg['cars'][0]]), simTime])
            time     = list(range(simTime))
            if FlagCreateGif:
                for t in time:
                    imageList.append(plotCurrentTimeGreedy(t,gridSize,lg))
                kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
                imageio.mimsave('./gif' +pickleName+ '.gif', imageList, fps=1)
                plt.close()
            plotBasicStatisticsOfEvents(gridSize, lg, pickleName, simTime)
            plotCarsHeatmap(gridSize,lg,simTime,pickleName)
            numEventsClosed = len([e for e in events.values() if e['closed'] and e['timeStart']+e['waitTime']<=simTime])
            print('number of closed events:'+str(numEventsClosed))
            cost = lg['cost']
            print('total cost is : '+str(cost))
        elif 'SimAnticipatory' in pickleName or 'Greedy' in pickleName:
            if FlagCreateGif:
                if not os.path.isdir(fileLoc + pickleName):
                    os.mkdir(fileLoc + pickleName)
            # this is the anticipatory results for inner MIO opt.
            time           = lg['time']
            gridSize       = lg['gs']
            simTime        = np.max(time)
            openedEvents   = np.array(lg['OpenedEvents'])
            closedEvents   = np.array(lg['closedEvents'])
            canceledEvnets = np.array(lg['canceledEvents'])
            allEvents      = np.array(lg['allEvents'])
            cost           = lg['cost']
            eventsPos      = [c.position for c in lg['pathresults'][-1].events.notCommited.values()]
            carsPos        = [c.path for c in lg['pathresults'][-1].cars.notCommited.values()]
            if FlagCreateGif:
                for t in time:
                    plotCurrentTimeAnticipatory(lg['pathresults'][t], len(eventsPos), len(carsPos), gridSize, fileLoc + '/' + pickleName + '/' + pickleName)
                listNames = [pickleName+'_'+str(t)+'.png' for t in time]
                create_gif(fileLoc+pickleName+'/', listNames, 1, pickleName)
            plotBasicStatisticsOfEvents(gridSize, lg, pickleName, simTime, fig, ax)
            # plotCarsHeatmap(gridSize, lg, simTime, pickleName)
            print('number of closed events:' + str(closedEvents[-1]))
            cost           = lg['cost']
            print('total cost is : ' + str(cost))
        elif 'Optimization' in pickleName:
            time            = lg['time']
            gridSize        = lg['gs']
            simTime         = np.max(time)
            closedEvents    = np.array(lg['closedEvents'])
            cost            = lg['cost']
            print('number of closed events:' + str(closedEvents[-1]))
            print('total cost is :'+str(cost))
            plotBasicStatisticsOfEvents(gridSize, lg, pickleName, simTime, fig, ax)

    plt.show()

if __name__ == main():
    main()
    print('done')