import  numpy as np
import  pickle
from matplotlib import pyplot as plt
import sys
import seaborn as sns
import imageio


sys.path.insert(0, '/home/chanaby/Documents/Thesis/aima-python')
from search import Problem,astar_search

sns.set()

def filterEvents(events,time):
    filteredEvents = [e for e in events if e[2]<=0 and e[3] == False]
    return filteredEvents


def plotCurrentTime(time,gridSize,cars,events):
    fig, ax = plt.subplots()
    # plot car positions
    for i,cid in enumerate(cars):
        pos = cid[1]
        ax.scatter(pos[0], pos[1], c='r', alpha=0.5, label='cars')
        ax.text(pos[0], pos[1] + 0.1, 'cid: {0}'.format(i))
    # plot event positions
    filteredEvents = filterEvents(events, time)
    for i,fev in enumerate(filteredEvents):
        ePos = fev[1]
        ax.scatter(ePos[0], ePos[1], c='b', alpha=0.5, label='events')
        ax.text(ePos[0], ePos[1] + 0.1, 'opens in: {0}'.format(fev[2]))
    for i,fev in enumerate(events):
        if fev not in filteredEvents and fev[3]==False:
            ePos = fev[1]
            ax.scatter(ePos[0], ePos[1], c='m', alpha=0.2, label='events not opened')
            ax.text(ePos[0], ePos[1] + 0.1, 'opens in: {0}'.format(fev[2]))
    ax.set_title('current time: {0}'.format(time))
    ax.grid(True)
    ax.legend()
    ax.set_ylim([-3, gridSize + 3])
    ax.set_xlim([-3, gridSize + 3])
    # plt.show()

    # Used to return the plot as an image rray
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

def plotEventOpenedTime(eventTimes,pickleName):
    eventId = range(len(eventTimes))
    plt.bar(eventId,eventTimes,label=pickleName,alpha = 0.7)
    plt.xlabel('event ID')
    plt.ylabel('Time event opened')
    plt.legend()
    plt.title('time event opened')





def main():
    #  load logs
    pickleNames = []
    pickleNames.append('aStarResult_1weight_4grid_8numEvents_8simTime_2numCars')
    pickleNames.append('aStarResult_2weight_4grid_8numEvents_8simTime_2numCars')
    pickleNames.append('aStarResult_5weight_4grid_8numEvents_8simTime_2numCars')
    pickleNames.append('aStarResult_10weight_4grid_8numEvents_8simTime_2numCars')
    imageList = []
    gridSize = 5
    FlagCreateGif = 0
    for pickleName in pickleNames:
        lg = pickle.load(open('/home/chanaby/Documents/Thesis/SearchAlgorithm//' + pickleName + '.p', 'rb'))
        timeSteps = []
        for temp in lg['shortestPath']:
            if temp[-1] == 1:
                timeSteps.append(temp)
        for i, t in enumerate(timeSteps):
            cars = []
            events = []
            for temp in t:
                if isinstance(temp, tuple) and 'car' in temp:
                    cars.append(temp)
                elif isinstance(temp, tuple) and 'event' in temp:
                    events.append(temp)
            if FlagCreateGif:
                imageList.append(plotCurrentTime(i, gridSize, cars, events))
        if FlagCreateGif:
            kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
            imageio.mimsave('./' + pickleName.replace('log', 'gif') + '.gif', imageList, fps=1)
            plt.close('all')
        plt.figure(88)
        eventTime = [-e[2] for e in events]
        totalEventTime = np.sum(eventTime)
        plotEventOpenedTime(eventTime, pickleName)
        pathCost = lg['aimaSolution'].path_cost
        totalPathTime = len(timeSteps)
        print(pickleName+' total path time:'+str(totalPathTime) + ' , total path cost:' + str(pathCost)+' ,total wait Time:'+str(totalEventTime))

    plt.show()


if __name__ == '__main__':
    main()
    print('finished')





