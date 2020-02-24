import imageio
from torch_geometric.data import DataLoader
# import my file in order to load state class from pickle
from Simulation.Anticipitory.with_RL.simAnticipatoryWithMIO_poisson_RL import *
from UtilsCode.createGif import create_gif
from MixedIntegerOptimization.offlineOptimizationProblem_TimeWindow import runMaxFlowOpt, plotResults
from RL_anticipatory.utils import load_model
from RL_anticipatory.problems.problem_anticipatory import AnticipatoryTestDataset
sns.set()
#  load logs
pickle_names = []

# test results on new code
# pickle_names.append('SimAnticipatoryMio_RandomChoice_5lpred_5delOpen_48startTime_20gridX_20gridY_31numEvents_20nStochastic_4numCars_Bm_poisson_TimeWindow')
# pickle_names.append('SimGreedy_5lpred_5delOpen_48startTime_20gridX_20gridY_31numEvents_20nStochastic_4numCars_Bm_poisson_TimeWindow')
# pickle_names.append('SimOptimization_MaxFlow_5lpred_5delOpen_48startTime_20gridX_20gridY_31numEvents_20nStochastic_4numCars_Bm_poisson_TimeWindow')
# pickle_names.append('SimOptimization_TimeWindow_5lpred_5delOpen_48startTime_20gridX_20gridY_31numEvents_20nStochastic_4numCars_Bm_poisson_TimeWindow')


# comparison to RL -
# pickle_names.append('SimAnticipatoryMio_RandomChoice_5lpred_5delOpen_48startTime_15gridX_15gridY_9numEvents_20nStochastic_2numCars_Bm_poisson_TimeWindow')
# pickle_names.append('SimGreedy_5lpred_5delOpen_48startTime_15gridX_15gridY_9numEvents_20nStochastic_2numCars_Bm_poisson_TimeWindow')
# pickle_names.append('SimOptimization_MaxFlow_5lpred_5delOpen_48startTime_15gridX_15gridY_9numEvents_20nStochastic_2numCars_Bm_poisson_TimeWindow')
# pickle_names.append('SimOptimization_TimeWindow_5lpred_5delOpen_48startTime_15gridX_15gridY_9numEvents_20nStochastic_2numCars_Bm_poisson_TimeWindow')


# comparison to RL -
pickle_names.append('SimAnticipatoryMio_RandomChoice_5lpred_5delOpen_48startTime_15gridX_15gridY_4numEvents_20nStochastic_2numCars_Bm_poisson_TimeWindow')
pickle_names.append('SimGreedy_5lpred_5delOpen_48startTime_15gridX_15gridY_4numEvents_20nStochastic_2numCars_Bm_poisson_TimeWindow')
pickle_names.append('SimOptimization_MaxFlow_5lpred_5delOpen_48startTime_15gridX_15gridY_4numEvents_20nStochastic_2numCars_Bm_poisson_TimeWindow')
pickle_names.append('SimOptimization_TimeWindow_5lpred_5delOpen_48startTime_15gridX_15gridY_4numEvents_20nStochastic_2numCars_Bm_poisson_TimeWindow')

network_names = []
network_names.append('anticipatory_rl_15/anticipatory_rl_20200221T084826/epoch-13.pt')
network_names.append('anticipatory_rl_15/anticipatory_rl_20200221T145415/epoch-199.pt')

def filter_events(eventDict, currentTime, lg):
    filterd_event_dict = {}
    for event in eventDict.values():
        current_status_log = [s for s in event['statusLog'] if float(s[0]) == currentTime][0]
        if current_status_log[1]:
            filterd_event_dict[event['id']] = event
    return filterd_event_dict

def manhattan_path(position1, position2):
    """
    calculates grid deltas between two positions relative to
    first position
    :param position1:
    :param position2:
    :return: dx (int), dy (int)
    """
    dx = position2[0] - position1[0]
    dy = position2[1] - position1[1]
    return dx, dy

def manhattan_dist(position1, position2):
    """
    calc manhatten distacne between two positions
    :param position1:
    :param position2:
    :return: distance (int)
    """
    dx,dy = manhattan_path(position1, position2)
    return float(abs(dx) + abs(dy))

def plot_current_time_greedy(time, gridSize, lg):

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
    filteredEvents = filter_events(lg['events'], time, lg)
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


def plot_cars_heatmap(gridSize, lg, simTime, pickleName):
    heat_mat = np.zeros(shape=(gridSize[0], gridSize[1]))
    if 'SimAnticipatory' in pickleName or 'Greedy' in pickleName:
        cars_pos = [c.path for c in lg['pathresults'][-1].cars.notCommited.values()]
        for car_pos in cars_pos:
            for pos in car_pos:
                heat_mat[int(pos[0]),int(pos[1])] +=1
            plt.figure()
            sns.heatmap(heat_mat)
            plt.title('Heat map of car location - anticipatory')
    elif 'Hungarian' in pickleName:
        cars = lg['cars']
        for car in cars.values():
            car_position = [c['position'] for c in car if c['time']<=simTime]
            for pos in car_position:
                heat_mat[int(pos[0]),int(pos[1])] +=1
            plt.figure()
            sns.heatmap(heat_mat)
            plt.title('Heat map of car location - greedy')


def plot_basic_statistics_of_events(gridSize, lg, pickle_name, simTime, fig, ax):
    plot_all_events = False
    if 'Hungarian' in pickle_name or 'Greedy' in pickle_name:
        label_str = 'Greedy Algorithm'
        line_style = '--'
        m_str="o"
    elif 'SimAnticipatory' in pickle_name and 'Bm_easy' in pickle_name:
        # label_str = 'Anticipatory Algorithm based on Benchmark - seq'
        label_str = 'Anticipatory Algorithm - Prior Knowledge'
        line_style = '-'
        plot_all_events = True
        m_str="s"
    elif 'SimAnticipatory' in pickle_name and 'NN' in pickle_name:
        # label_str = 'Anticipatory Algorithm based on NN'
        label_str = 'Anticipatory Algorithm - NN'
        line_style = '-'
        plot_all_events = True
        m_str="s"
    elif 'SimAnticipatory' in pickle_name and 'Bm' in pickle_name:
        # label_str = 'Anticipatory Algorithm based on Benchmark'
        label_str = 'Anticipatory Algorithm - Prior Knowledge'
        line_style = '-'
        plot_all_events = True
        m_str="s"
    elif 'Optimization' in pickle_name and 'TimeWindow' in pickle_name:
        # label_str = 'determinsitc MIO results with time window'
        label_str = 'MIO - with time window'
        line_style = ':'
        m_str ="d"
    elif 'Optimization' in pickle_name and 'MaxFlow' in pickle_name:
        # label_str = 'determinsitc MIO results - max flow'
        label_str = 'MIO - max flow'
        line_style = ':'
        m_str='d'
    if 'SimAnticipatory' in pickle_name or 'Greedy' in pickle_name or 'Optimization' in pickle_name:
        if plot_all_events:
            ax.plot(lg['time'], lg['allEvents'], c='k', marker='*', markersize=8, linewidth=1.5, label='Num Total Events')
            plot_all_events = False
        ax.plot(lg['time'], lg['closedEvents'], linestyle=line_style, linewidth=1.5, marker=m_str, label=label_str)
        # plt.plot(lg['time'], lg['canceledEvents'], c='y', linestyle=line_style, label='canceled')
        ax.legend(loc='upper left', fontsize="medium")
        ax.grid(True)
        ax.set_xlabel('time', fontsize=20)
        ax.set_ylabel('num events', fontsize=20)
        ax.set_title('number of events over time', fontsize=20)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)

        # current over time
        plt.figure(3)
        plt.plot(lg['time'], lg['OpenedEvents'], label='current for :' + label_str)
        plt.title('currently open over time')
        plt.xlabel('time')
        plt.ylabel('num events')
        plt.grid(True)
        plt.legend()
        if 'Optimization' not in pickle_name:
            events_pos = [c.position for c in lg['pathresults'][-1].events.notCommited.values()]
            events_start_time = [c.startTime for c in lg['pathresults'][-1].events.notCommited.values()]
            events_end_time = [c.endTime for c in lg['pathresults'][-1].events.notCommited.values()]
            events_status = [c.status for c in lg['pathresults'][-1].events.notCommited.values()]
            plt.figure(4)
            for (i,ePos,eStatus) in zip(range(len(events_pos)), events_pos, events_status):
                if eStatus == Status.CLOSED:
                    plt.scatter(ePos[0],ePos[1], color='g', label=str(i))
                    plt.text(ePos[0], ePos[1], i)
                else:
                    plt.scatter(ePos[0], ePos[1], color='r', label=str(i))
                    plt.text(ePos[0], ePos[1], i)
            plt.xlabel('grid X')
            plt.ylabel('grid Y')
            plt.title('event locations for: ' + label_str)
            plt.xlim([-3, gridSize[0] + 3])
            plt.ylim([-3, gridSize[1] + 3])

    if 'Hungarian' in pickle_name:
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
        plt.plot(plotingTimeline, plotClosed, label='Num Closed for :'+label_str)
        plt.plot(plotingTimeline, eventLog['canceled'], c='y', label='canceled')
        plt.legend()
        plt.grid(True)
        plt.xlabel('time')
        plt.ylabel('num events')
        plt.title('number of events over time')

        # current over time
        plt.figure(3)
        plt.plot(plotingTimeline,plotCurrent , label='current for :' + label_str)
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
        plt.title('event locations for: ' + label_str)
        plt.xlim([-3, gridSize + 3])
        plt.ylim([-3, gridSize + 3])


def plot_current_time_anticipatory(s, ne, nc, gs, fileName):
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
        car_temp = s.cars.getObject(c)
        ax.scatter(car_temp.position[0], car_temp.position[1], c='k', alpha=1, marker='s')
    ax.scatter([], [], c='y', label='Future Requests')
    ax.scatter([], [], c='b', label='Opened')
    # ax.scatter([], [], c='b', label='Opened commited')
    ax.scatter([], [], c='r', label='Canceled')
    ax.scatter([], [], c='g', label='Closed')
    for i in range(ne):
        event_temp = s.events.getObject(i)
        if event_temp.status == Status.OPENED_COMMITED:
            ax.scatter(event_temp.position[0], event_temp.position[1], c='b', alpha=0.8)
        elif event_temp.status == Status.OPENED_NOT_COMMITED:
            ax.scatter(event_temp.position[0], event_temp.position[1], c='b', alpha=1)
        elif (event_temp.status == Status.CLOSED):
            ax.scatter(event_temp.position[0], event_temp.position[1], c='g', alpha=0.6)
        elif (event_temp.status == Status.CANCELED):
            ax.scatter(event_temp.position[0], event_temp.position[1], c='r', alpha=0.6)
        else:
            ax.scatter(event_temp.position[0], event_temp.position[1], c='y', alpha=0.7, marker='+')
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


def optimized_simulation(initialState, numFigure):
    cars_pos         = np.zeros(shape=(initialState.cars.length(), 2))
    events_pos       = []
    events_start_time = []
    events_end_time   = []
    for d, k in enumerate(initialState.cars.getUnCommitedKeys()):
        cars_pos[d, :] = deepcopy(initialState.cars.getObject(k).position)
    # get opened event locations from state -
    for k in initialState.events.getUnCommitedKeys():
        events_pos.append(deepcopy(initialState.events.getObject(k).position))
        events_start_time.append(deepcopy(initialState.events.getObject(k).startTime))
        events_end_time.append(deepcopy(initialState.events.getObject(k).endTime))

    m, obj = runMaxFlowOpt(0, cars_pos, np.array(events_pos), np.array(events_start_time), np.array(events_end_time), initialState.closeReward,
                           initialState.cancelPenalty, initialState.openedNotCommitedPenalty, 0)
    plotResults(m, cars_pos, np.array(events_pos), np.array(events_start_time), np.array(events_end_time), numFigure)


def main():
    image_list = []
    # model parameters -
    flag_calc_network = True
    if flag_calc_network:
        seed = 12
        torch.manual_seed(seed)
        network_loc = '/Users/chanaross/dev/Thesis/RL_anticipatory/outputs/'
    # general parameters -
    flag_create_gif = 0
    file_loc = '/Users/chanaross/dev/Thesis/Simulation/Anticipitory/with_RL/Results/'
    fig, ax = plt.subplots(1, 1)
    for pickle_name in pickle_names:
        lg  = pickle.load(open(file_loc + pickle_name + '.p', 'rb'))
        sim_time = 20
        if 'Hungarian' in pickle_name:
            events   = lg['events']
            grid_size = lg['gs']
            sim_time  = np.min([np.max([c['time'] for c in lg['cars'][0]]), sim_time])
            time     = list(range(sim_time))
            if flag_create_gif:
                for t in time:
                    image_list.append(plot_current_time_greedy(t, grid_size, lg))
                kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
                imageio.mimsave('./gif' +pickle_name+ '.gif', image_list, fps=1)
                plt.close()
            plot_basic_statistics_of_events(grid_size, lg, pickle_name, sim_time, fig, ax)
            plot_cars_heatmap(grid_size, lg, sim_time, pickle_name)
            numEventsClosed = len([e for e in events.values() if e['closed'] and e['timeStart']+e['waitTime']<=sim_time])
            print('number of closed events:'+str(numEventsClosed))
            cost = lg['cost']
            print('total cost is : '+str(cost))
        elif 'SimAnticipatory' in pickle_name or 'Greedy' in pickle_name:
            if flag_create_gif:
                if not os.path.isdir(file_loc + pickle_name):
                    os.mkdir(file_loc + pickle_name)
            # this is the anticipatory results for inner MIO opt.
            time           = lg['time']
            grid_size       = lg['gs']
            sim_time        = np.max(time)
            opened_events   = np.array(lg['OpenedEvents'])
            closed_events   = np.array(lg['closedEvents'])
            canceled_events = np.array(lg['canceledEvents'])
            allEvents      = np.array(lg['allEvents'])
            cost           = lg['cost']
            events_pos      = [c.position for c in lg['pathresults'][-1].events.notCommited.values()]
            cars_pos        = [c.path for c in lg['pathresults'][-1].cars.notCommited.values()]
            if flag_create_gif:
                for t in time:
                    plot_current_time_anticipatory(lg['pathresults'][t], len(events_pos), len(cars_pos), grid_size, file_loc + '/' + pickle_name + '/' + pickle_name)
                list_names = [pickle_name+'_'+str(t)+'.png' for t in time]
                create_gif(file_loc+pickle_name+'/', list_names, 1, pickle_name)
            plot_basic_statistics_of_events(grid_size, lg, pickle_name, sim_time, fig, ax)
            # plotCarsHeatmap(gridSize, lg, simTime, pickleName)
            print('number of closed events:' + str(closed_events[-1]))
            cost           = lg['cost']
            print('total cost is : ' + str(cost))
            if 'SimAnticipatory' in pickle_name:
                events_times = lg['event_times']
                car_starting_loc = lg['car_loc']
        elif 'Optimization' in pickle_name:
            time            = lg['time']
            grid_size        = lg['gs']
            sim_time         = np.max(time)
            closed_events    = np.array(lg['closedEvents'])
            cost            = lg['cost']
            print('number of closed events:' + str(closed_events[-1]))
            print('total cost is :'+str(cost))
            plot_basic_statistics_of_events(grid_size, lg, pickle_name, sim_time, fig, ax)
    if flag_calc_network:
        states = []
        for i_m, network_name in enumerate(network_names):
            model, args, sim_input, stochastic_input = load_model(network_loc + network_name)
            if i_m == 0:
                data_input = {'car_loc': car_starting_loc,
                              'events_time': events_times,
                              'events_loc': events_pos,
                              'cancel_cost': args['cancel_cost'],
                              'close_reward': args['close_reward'],
                              'movement_cost': args['movement_cost'],
                              'open_cost': args['open_cost'],
                              'n_cars': args['n_cars'],
                              'end_time': sim_input['sim_length'],
                              'lam': args['lam'],
                              'events_time_window': sim_input['events_open_time'],
                              'graph_size': args['graph_size']}
                data = AnticipatoryTestDataset(root="", data_input=data_input)
                dataloader = DataLoader(data, batch_size=1)
                batch_data = next(iter(dataloader))
            model.eval()
            model.set_decode_type('sampling')
            with torch.no_grad():
                costs_all_options, logits_all_options, actions_chosen, logits_chosen, cost_chosen, state = model(batch_data)
            states.append(state)
        print(cost)
    plt.show()


if __name__ == main():
    main()
    print('done')