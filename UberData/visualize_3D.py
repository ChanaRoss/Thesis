import numpy as np
import pandas as pd
from matplotlib import  pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(3.5,11)})
import sys
sys.path.insert(0, '/Users/chanaross/dev/Thesis/UtilsCode/')
from createGif import create_gif





def plotSpesificTime(data, t, fileName):
    hour,temp = np.divmod(t, 12)
    hour += 8   # data starts from 98 but was normalized to 0
    minute = temp*5
    dataFixed = np.zeros_like(data)
    dataFixed = np.swapaxes(data, 1, 0)
    dataFixed = np.flipud(dataFixed)
    sns.heatmap(dataFixed, cbar = True, center = 1,square=True, vmin = 0, vmax = 4, cmap = 'CMRmap_r', cbar_kws=dict(ticks=[0, 1, 2, 3, 4]))
    plt.title('time is -  {0}:{1}'.format(hour, minute))
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.savefig(fileName + '_' + str(t) +'.png')
    plt.close()
    return

#Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu,
# GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1,
# Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples,
# Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r,
# Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r,
# YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis,
# cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth,
# gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r,
# gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r,
# hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral,
# nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r,
# seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c,
# tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, vlag,
# vlag_r, winter, winter_r



def main():

    path = '/Users/chanaross/dev/Thesis/UberData/'
    data_name = '3D_allDataLatLonCorrected_250gridpickle_5min.p'

    data = pd.read_pickle(path + data_name)

    for t in range(300):  #range(data.shape[2]):
        plotSpesificTime(data[:, :, t], t, path + 'picsFor3D/dataOut')
    listNames = ['dataOut' + '_' + str(t) + '.png' for t in range(300)]
    create_gif(path + 'picsFor3D/', listNames, 1, 'dataOut')
    return





if __name__ == main():
    main()
    print('done')