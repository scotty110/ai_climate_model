import torch

from os.path import join

import utils


if __name__ == '__main__':
    fname = join('/home/squirt/Documents/data/rp_weather_data/', 'all_data.h5')
    t_dl, v_dl = utils.get_dataloaders(fname, 256, 0.8)

    for i, (l,x,y) in enumerate(t_dl):
        print(l.shape, x.shape, y.shape)
        if i > 5:
            break