# mathematical imports -
import numpy as np
import scipy.stats as stats
# graphical imports -
from matplotlib import pyplot as plt
import seaborn as sns

import os,sys
sys.path.insert(0, '/Users/chanaross/dev/Thesis/UtilsCode/')
from createGif import create_gif

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


def calcAverageResults(cdf_mat, timeIndexs):
    y_out = np.zeros([timeIndexs.size, 1])
    for t in timeIndexs:
        randNum = np.random.uniform(0, 1)
        # find how many events are happening at the same time
        cdf = cdf_mat[t, :].reshape(cdf_mat.shape[1])
        numEvents = np.searchsorted(cdf, randNum, side='left')
        numEvents = np.floor(numEvents).astype(int)
        y_out[t] = numEvents
    return y_out


def create_gif_plot(mat, nweek_plot, num_days, num_ticks_per_day, pathName, fileName):
    # dataFixed = np.zeros_like(mat)
    # dataFixed = np.swapaxes(mat, 1, 0)
    # dataFixed = np.flipud(dataFixed)
    dataFixed = mat
    max_ticks = np.max(dataFixed).astype(int)
    ticksDict = list(range(max_ticks + 1))
    fig, axes = plt.subplots(1, 1)
    t = 0

    for nday in range(num_days):
        for nticks in range(num_ticks_per_day):
            mat_plot = dataFixed[:, :, nticks, nday, nweek_plot]
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



def main():
    np.random.seed(111)
    inner_prob = 3
    outer_prob = 1
    num_x     = inner_prob + 2*outer_prob  # size of y axis
    num_y     = inner_prob + 2*outer_prob  # size of x axis

    x_plot    = 0
    y_plot    = 0
    nweek_plot= 0

    num_ticks_per_day = 24*1
    num_days  = 7
    num_weeks = 50
    y_out     = np.zeros([num_x, num_y, num_ticks_per_day, num_days, num_weeks])
    mat_out   = np.zeros([num_x, num_y, num_ticks_per_day*num_days*num_weeks])
    y_for_prob = np.zeros([num_x, num_y, num_ticks_per_day*num_days, num_weeks])
    color_plt = np.random.rand(3, 20)
    sigma     = 3
    start_mu  = 2
    small_coef = 0.1
    large_coef = 0.9
    fig_path = '/Users/chanaross/dev/Thesis/ProbabilityFunction/theoreticalProbability/figures/'
    fig_name = 'gif_theoretical_dist'
    m        = ['*', '.', 'o', 's', '<']
    time_axes = np.arange(0, num_ticks_per_day)
    for x in range(num_x):
        for y in range(num_y):
            k = 0
            if (outer_prob <= x < outer_prob + inner_prob) and (outer_prob <= y < outer_prob + inner_prob):
                # in inner area, therefore taxi's are needed in the morning
                coef_a = small_coef
                coef_b = large_coef
            else:
                # in inner area, taxi's are needed in the evening
                coef_a = large_coef
                coef_b = small_coef

            for nweek in range(num_weeks):
                k_prob  = 0
                min_val = 1
                max_val = 5 + (nweek*0.1)*1.1
                mu = start_mu + (nweek*0.1)*1.1
                dist        = stats.truncnorm((min_val - mu) / sigma, (max_val - mu) / sigma, loc=mu, scale=sigma)
                amp_per_day = dist.rvs(num_days)  # np.random.normal(mu, sigma, num_days)
                for i in range(num_days):
                    for j in range(num_ticks_per_day):
                        if j<num_ticks_per_day*0.5:
                            y_out[x, y, j, i, nweek] = np.floor(coef_a*np.abs(amp_per_day[i]*np.sin(2*np.pi*j/num_ticks_per_day)))
                        else:
                            y_out[x, y, j, i, nweek] = np.floor(coef_b*np.abs(amp_per_day[i] * np.sin(2 * np.pi * j / num_ticks_per_day)))
                        mat_out[x, y, k] = y_out[x, y, j, i, nweek]
                        y_for_prob[x, y, k_prob] = y_out[x, y, j, i, nweek]
                        k += 1
                        k_prob += 1
                    if nweek == 0 and x == x_plot and y == y_plot:
                        plt.plot(time_axes + num_ticks_per_day*i + nweek*num_ticks_per_day*num_days, y_out[x, y, :, i, nweek],
                                 marker = m[nweek], label='wday - '+str(i), color=color_plt[:, i])
                    elif x == x_plot and y == y_plot:
                        plt.plot(time_axes + num_ticks_per_day * i + nweek * num_ticks_per_day * num_days, y_out[x, y, :, i, nweek],
                                 marker=m[nweek%4], color=color_plt[:, i])
    mat_prob, cdf_prob = createProbabilityMatrix(y_for_prob)
    # mat_prob.dump('Probablity_4Dmat_theoreticalData_' +str(start_mu)+'_mu_'+str(sigma) +'_sigma'  + str(small_coef)+'_smallCoeff'+ '.p')
    # cdf_prob.dump('CDF_4Dmat_theoreticalData_' +str(start_mu)+'_mu_'+str(sigma) +'_sigma' + str(small_coef)+'_smallCoeff' + '.p')
    # mat_out.dump('3Dmat_theoreticalData_' +str(start_mu)+'_mu_'+str(sigma) +'_sigma_' + str(small_coef)+'_smallCoeff'+ '.p')

    plt.grid()
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Num Events')
    plt.show()
    # plt.close()
    #
    # create_gif_plot(y_out, nweek_plot, num_days, num_ticks_per_day, fig_path, fig_name)






if __name__ == '__main__':
    main()
    print('Done.')