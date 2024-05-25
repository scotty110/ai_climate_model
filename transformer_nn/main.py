import torch

from os.path import join

import utils

# Use cuda (else throw error, running on GPU/Grace Hopper)
assert torch.cuda.is_available(), 'CUDA is not available.'


if __name__ == '__main__':
    fname = join('/home/squirt/Documents/data/weather_data/', 'all_data.h5')

    # Going to cross validate (since there is no real test set)
    t_dl, v_dl = utils.get_dataloaders(fname, 256, 0.8)

    # Model 
    out_dim = int(70*2)
    model = utils.ConvTrans()

    for i, (l,x,y) in enumerate(t_dl):
        print(l.shape, x.shape, y.shape)

        x = x.float()
        print(model(x).shape)
        if i > 5:
            break