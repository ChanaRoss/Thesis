# mathematical imports -
import numpy as np
from scipy.stats import truncnorm


# plotting imports -
from matplotlib import pyplot as plt
import seaborn as sns

# system imports
import os, sys

# my functions
sys.path.insert(0, '/Users/chanaross/dev/Thesis/UtilsCode/')
from createGif import create_gif

sns.set()

class Customer:
    def __init__(self, time, location, index):
        self.time     = np.array(time)
        self.location = np.array(location)
        self.id       =  index

    def add_customer(self, event_mat, event_mat_split, event_mat_for_prob, num_steps_per_day, day, week):
        for t, loc in zip(self.time, self.location):
            total_day = week*7 + day
            event_mat[loc[0], loc[1], num_steps_per_day*total_day + (t-1)] += 1
            event_mat_split[loc[0], loc[1], t-1, day, week] += 1
            event_mat_for_prob[loc[0], loc[1], num_steps_per_day*day + (t-1), week] += 1
        return event_mat, event_mat_split, event_mat_for_prob


def get_loc(loc_options):
    index = np.random.randint(0, len(loc_options))
    home_loc = loc_options[index]
    return home_loc


def create_home_work_list(grid_size_x, grid_size_y):
    list_work   = []
    list_home   = []
    list_outing = []
    work_base = []
    mat_locations = np.zeros([grid_size_x, grid_size_y])

    for i in range(2):
        # create work main locations.
        # around point chosen as center there is a 70% chance of offices and 30% chance of outing
        work_x = np.random.randint(0, grid_size_x)
        work_y = np.random.randint(0, grid_size_y)
        count = 0
        if mat_locations[work_x, work_y] ==3:
            is_taken = True
        else:
            is_taken = False
        while (is_taken):
            if count >= grid_size_y*grid_size_x:
                is_taken = False
            elif mat_locations[work_x, work_y] == 3:
                is_taken = True
                count += 1
                work_x = np.random.randint(0, grid_size_x)
                work_y = np.random.randint(0, grid_size_y)
        mat_locations[work_x, work_y] = 3
        if (work_x + 1 < grid_size_x):
            rand_num_w = np.random.uniform(0, 1)
            if rand_num_w < 0.7:
                mat_locations[work_x + 1, work_y] = 1
            else:
                mat_locations[work_x + 1, work_y] = 2

            if (work_y + 1 < grid_size_y):
                rand_num_w = np.random.uniform(0, 1)
                if rand_num_w < 0.7:
                    mat_locations[work_x+1, work_y+1] = 1
                else:
                    mat_locations[work_x+1, work_y+1] = 2
            if (work_y - 1 >= 0):
                rand_num_w = np.random.uniform(0, 1)
                if rand_num_w < 0.7:
                    mat_locations[work_x + 1, work_y - 1] = 1
                else:
                    mat_locations[work_x + 1, work_y - 1] = 2
        if (work_x - 1 >= 0):
            rand_num_w = np.random.uniform(0, 1)
            if rand_num_w < 0.7:
                mat_locations[work_x - 1, work_y] = 1
            else:
                mat_locations[work_x - 1, work_y] = 2
            if (work_y + 1 < grid_size_y):
                rand_num_w = np.random.uniform(0, 1)
                if rand_num_w < 0.7:
                    mat_locations[work_x-1, work_y+1] = 1
                else:
                    mat_locations[work_x-1, work_y+1] = 2
            if (work_y - 1 >= 0):
                rand_num_w = np.random.uniform(0, 1)
                if rand_num_w < 0.7:
                    mat_locations[work_x - 1, work_y - 1] = 1
                else:
                    mat_locations[work_x - 1, work_y - 1] = 2

        if (work_y + 1 < grid_size_y):
            rand_num_w = np.random.uniform(0, 1)
            if rand_num_w < 0.7:
                mat_locations[work_x, work_y + 1] = 1
            else:
                mat_locations[work_x, work_y + 1] = 2
        if (work_y -1 >= 0):
            rand_num_w = np.random.uniform(0, 1)
            if rand_num_w < 0.7:
                mat_locations[work_x, work_y - 1] = 1
            else:
                mat_locations[work_x, work_y - 1] = 2
        work_base.append(np.array([work_x, work_y]))
        sns.heatmap(mat_locations, vmin=0, vmax=3, cbar=True, center=1, square=True)
        plt.savefig('location_heatmap.png')
        plt.close()
        plt.scatter([], [], color='k', label='work')
        plt.scatter([], [], color='m', label='home')
        plt.scatter([], [], color='g', label='outing')
        plt.scatter([], [], color='b', marker='*', label='work_center')
        plt.scatter(work_x, work_y, color = 'b', marker='*')
    print(mat_locations)
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            if mat_locations[i, j] == 1 or mat_locations[i, j] == 3:
                list_work.append(np.array([i, j]))
                plt.scatter(i, j, color='k')
            elif mat_locations[i, j] == 2:
                list_outing.append(np.array([i, j]))
                plt.scatter(i, j, color='g')
            else:
                list_home.append(np.array([i, j]))
                plt.scatter(i, j, color='m')
            # rand_num = np.random.uniform(0, 1)
            # if rand_num < 0.1:
            #     list_work.append(np.array([i, j]))
            #     plt.scatter(i, j, color='k')
            # elif rand_num < 0.4:
            #     list_home.append(np.array([i, j]))
            #     plt.scatter(i, j, color='m')
            # elif rand_num < 0.6:
            #     list_outing.append(np.array([i, j]))
            #     plt.scatter(i, j, color='g')
    plt.legend()
    plt.savefig('location_fig.png')
    plt.close()
    # plt.show()
    return list_home, list_work, list_outing, work_base



def create_customer(id, num_steps_per_day, list_home, list_work, list_outing, distributions):
    time     = []
    location = []

    prob_outing        = 0.5  # percentage of customers that go out after work
    prob_direct_outing = 0.3  # percentage of outings that are direct (from the customers that go out)

    home_loc           = get_loc(list_home)
    work_loc           = get_loc(list_work)

    rand_outing        = np.random.uniform(0, 1)
    rand_direct_outing = np.random.uniform(0, 1)

    morning_time       = np.floor(distributions['morning_time'][id]).astype(int)
    morning_loc        = home_loc

    time.append(morning_time)
    location.append(morning_loc)

    evening_time       = np.floor(distributions['evening_time'][id]).astype(int)
    evening_loc        = work_loc

    time.append(evening_time)
    location.append(evening_loc)

    if rand_outing < prob_outing:
        outing_loc     = get_loc(list_outing)
        outing_length  = np.random.randint(1, 3)
        if rand_direct_outing < prob_direct_outing:
            end_time   = evening_time + outing_length
            end_loc    = outing_loc
            time.append(end_time)
            location.append(end_loc)
        else:
            outing_start = np.random.randint(1, 3)
            outing_start_time = evening_time + outing_start
            time.append(outing_start_time)
            location.append(home_loc)  # going from home to outing

            outing_end_time   = np.min([outing_start_time + outing_length, 24])
            time.append(outing_end_time)
            location.append(outing_loc)

    customer = Customer(time, location, id)
    return customer

def create_gif_plot(mat, nweek_plot, num_days, num_ticks_per_day, pathName, fileName):
    max_ticks = np.max(mat).astype(int)
    ticksDict = list(range(max_ticks + 1))
    fig, axes = plt.subplots(1, 1)
    t = 0

    for nday in range(num_days):
        for nticks in range(num_ticks_per_day):
            mat_plot = mat[:, :, nticks, nday, nweek_plot]
            try:
                sns.heatmap(mat_plot, cbar=True, center=1, square=True, vmin=0, vmax=np.max(ticksDict),
                            cmap='CMRmap_r', cbar_kws=dict(ticks=ticksDict))
            except:
                print("hi")
            plt.suptitle('week- {0}, day- {1},time- {2}:00'.format(nweek_plot, nday, t), fontsize=16)
            plt.title('num events')
            plt.xlabel('X axis')
            plt.ylabel('Y axis')
            plt.savefig(pathName + fileName + '_' + str(t) + '.png')
            plt.close()
            t += 1

    timeIndexs = list(range(t))
    listNames = [fileName + '_' + str(t) + '.png' for t in timeIndexs]
    create_gif(pathName, listNames, 1, fileName)
    for fileName in listNames:
        os.remove(pathName + fileName)

def plot_graph_vs_time(mat, grid_loc, str):
    time = np.array(list(range(mat.shape[2])))
    plt.plot(time, mat[grid_loc[0], grid_loc[1], :], marker = '*', label = str)
    plt.legend()
    plt.savefig('time_plot_'+ str +'.png')
    plt.close()


def createProbabilityMatrix(inputMat):
    """

    :param inputMat: shape is [grid_x, grid_y, num_times_per_week, num_week]
    :return:
    """
    num_events  = np.max(inputMat).astype(int)
    mat         = np.zeros([inputMat.shape[0], inputMat.shape[1], inputMat.shape[2], num_events+1])
    mat_out     = np.zeros_like(mat)
    cdf         = np.zeros(num_events+1)
    cdf_out     = np.zeros_like(mat_out)
    for x in range(inputMat.shape[0]):
        for y in range(inputMat.shape[1]):
            for t in range(inputMat.shape[2]):
                for n_week in range(inputMat.shape[3]):
                    temp          = inputMat[x, y, t, n_week].astype(int)
                    mat[x, y, t, temp] += 1
                mat_out[x, y, t, :] = mat[x, y, t, :]/np.sum(mat[x, y, t, :])
                for j in range(num_events+1):
                    cdf[j] = np.sum(mat_out[x, y, t, 0:j + 1])
                cdf_out[x, y, t, :] = cdf
    return mat_out, cdf_out


def main():
    np.random.seed(10)
    num_customers = 60
    num_weeks     = 100
    num_days      = 7*num_weeks
    nweek_plot    = 0
    grid_size_x   = 6
    grid_size_y   = 6
    fig_path      = '/Users/chanaross/dev/Thesis/ProbabilityFunction/personBasedProbability/figures/'
    fig_name      = 'probabilityGif'
    # create time for morning ride -
    num_steps_per_day = 24
    list_home, list_work, list_outing, work_base = create_home_work_list(grid_size_x, grid_size_y)

    event_mat         = np.zeros([grid_size_x, grid_size_y, num_steps_per_day*num_days])
    event_mat_split   = np.zeros([grid_size_x, grid_size_y, num_steps_per_day, num_days, num_weeks])
    event_mat_for_prob = np.zeros([grid_size_x, grid_size_y, num_steps_per_day*7, num_weeks])

    id                = 0
    customer_dict = {}
    distributions = {}
    # going to work time -
    myclip_a    = 4
    myclip_b    = 11
    my_mean     = 8
    my_std      = 3
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    distributions['morning_time'] = truncnorm.rvs(a, b, loc=my_mean, scale=my_std, size=num_customers*num_days*num_weeks)
    # leaving work time -
    myclip_a    = 13
    myclip_b    = 21
    my_mean     = 17
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    distributions['evening_time'] = truncnorm.rvs(a, b, loc=my_mean, scale=my_std, size=num_customers*num_days*num_weeks)
    for nweek in range(num_weeks):
        for nday in range(7):
            for i in range(num_customers):
                customer = create_customer(id, num_steps_per_day, list_home, list_work, list_outing, distributions)
                customer_dict[id] = customer
                event_mat, event_mat_split, event_mat_for_prob = customer.add_customer(event_mat, event_mat_split,
                                                                                       event_mat_for_prob,
                                                                                       num_steps_per_day, nday, nweek)
                id += 1
    event_mat_prob, event_mat_cdf = createProbabilityMatrix(event_mat_for_prob)
    # create_gif_plot(event_mat_split, nweek_plot, num_days, num_steps_per_day, fig_path, fig_name)
    plot_graph_vs_time(event_mat, list_work[0], 'work_loc_'+str(num_weeks)+'_nweeks')
    plot_graph_vs_time(event_mat, list_home[0], 'home_loc_'+str(num_weeks)+'_nweeks')
    plot_graph_vs_time(event_mat, list_outing[0], 'outing_loc_'+str(num_weeks)+'_nweeks')
    event_mat.dump('3Dmat_personBasedData_' + str(num_customers) + '_numCustomers' + '.p')
    event_mat_prob.dump('Probablity_4Dmat_personBasedData_' + str(num_customers)+'_numCustomers' + '.p')
    event_mat_cdf.dump('CDF_4Dmat_personBasedData_' + str(num_customers) + '_numCustomers' + '.p')

    print("hi")








if __name__ == '__main__':
    main()
    print('Done.')

