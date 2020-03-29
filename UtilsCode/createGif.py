import os, sys
import datetime
import imageio
from pprint import pprint
import time
import datetime

e = sys.exit


def create_gif(fileLoc, filenames, duration, outName):
    images = []
    for filename in filenames:
        images.append(imageio.imread(fileLoc + filename))
    imageio.mimsave(fileLoc + outName + '.gif', images, duration=duration)


if __name__ == "__main__":
    script = sys.argv.pop(0)
    duration = 0.2
    time = list(range(24))
    filenames = ['testplot'+str(t)+'.png' for t in time]
    fileLoc = '/Users/chanaross/dev/Thesis/ProbabilityFunction/CreateEvents/TestGif/'
    # filenames = sorted(filter(os.path.isfile, [x for x in os.listdir() if x.endswith(".png")]),
    #                    key=lambda p: os.path.exists(p) and os.stat(p).st_mtime or time.mktime(
    #                        datetime.now().timetuple()))

    create_gif(fileLoc, filenames, duration)


