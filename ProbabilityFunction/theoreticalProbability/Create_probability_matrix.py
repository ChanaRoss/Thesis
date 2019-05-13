# mathematical imports -
import numpy as np
import scipy
# graphical imports -
from matplotlib import pyplot as plt
import seaborn as sns


def createProbabilityMatrix(inputMat):
    """

    :param inputMat: shape is [num_cases, num_time_steps, 1]
    :return:
    """
    num_events  = np.max(inputMat).astype(int)
    mat         = np.zeros([inputMat.shape[1], num_events+1])
    mat_out     = np.zeros_like(mat)
    cdf         = np.zeros(num_events+1)
    cdf_out     = np.zeros_like(mat_out)
    for t in range(inputMat.shape[1]):
        for i in range(inputMat.shape[0]):  # going through all the cases for time t
            temp          = inputMat[i, t, 0].astype(int)
            mat[t, temp] += 1
        mat_out[t, :] = mat[t, :]/np.sum(mat[t, :])
        for j in range(num_events+1):
            cdf[j] = np.sum(mat_out[t, 0:j + 1])
        cdf_out[t, :] = cdf
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


# def main():
#     sigma       = 1.2
#     num_ticks   = 24*7
#     num_cases   = 10
#
#     mu          = 3
#     y_out       = np.zeros([num_cases, num_ticks, 1])
#     t = np.arange(0, num_ticks, 1).reshape(num_ticks, 1)
#     fig, ax = plt.subplots(1, 1)
#     fig1, ax1 = plt.subplots(1, 1)
#     for i in range(num_cases):
#         amp = np.random.normal(mu, sigma, 1)[0]
#         y     = np.abs(np.floor(amp*(np.sin(np.pi*t/(24)))))   # + 4*amp*(np.sin(2*t*np.pi/(24*7)))))
#         for j in range(num_ticks):
#             val = np.random.normal(y[j], sigma, 1)[0]
#             y_out[i, j, :] = y[j]
#         # ax.plot(t+(t[-1])*i, y, marker='.')
#         ax.plot(t+(t[-1])*i, y_out[i, :, :], marker='.')
#         ax1.plot(t, y_out[i, :, :])
#     ax.grid()
#     ax1.grid()
#     y_average = np.mean(y_out, 0).reshape(num_ticks, 1)
#     probMat, cdfMat = createProbabilityMatrix(y_out)
#     y_prob_out = calcAverageResults(cdfMat, t)
#     # ax1.plot(t, y_average, color='k', linewidth=2, linestyle='-')
#     ax1.plot(t, y_prob_out, color='k', linewidth=2, linestyle='--')
#     # for i in range(num_cases):
#         # ax.plot(t + (t[-1] * i), y_average, color='k', linewidth=2)
#         # ax.plot(t + (t[-1] * i), y_prob_out, color='k', linewidth=2)
#
#     plt.show()
#     return



def main():
    num_ticks_per_day = 24*3
    num_days  = 7
    num_weeks = 1
    y_out     = np.zeros([num_ticks_per_day, num_days, num_weeks])
    mu        = 6
    sigma     = 2
    m         = ['*', '.', 'o']
    time_axes = np.arange(0, num_ticks_per_day)
    for nweek in range(num_weeks):
        amp_per_day = np.random.normal(mu, sigma, num_days)
        for i in range(num_days):
            for j in range(num_ticks_per_day):
                if j<num_ticks_per_day*0.5:
                    y_out[j, i, nweek] = np.floor(np.abs(amp_per_day[i]*np.sin(2*np.pi*j/num_ticks_per_day)))
                else:
                    y_out[j, i, nweek] = np.floor(0.5*np.abs(amp_per_day[i] * np.sin(2 * np.pi * j / num_ticks_per_day)))
            plt.plot(time_axes + num_ticks_per_day*i + nweek*num_ticks_per_day*num_days, y_out[:, i, nweek], marker = m[nweek], label = 'wday - '+str(i))
    plt.grid()
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Num Events')
    plt.show()





if __name__ == '__main__':
    main()
    print('Done.')