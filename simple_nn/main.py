import torch
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from os.path import join

import utils

# Use cuda (else throw error, running on GPU/Grace Hopper)
assert torch.cuda.is_available(), 'CUDA is not available.'

'''
Functions for Training
'''

class simpleNN(nn.Module):
    def __init__(self):
        super(simpleNN, self).__init__()
        self.fc1 = nn.Linear(852,4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 560)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model:nn.Module, dl:torch.utils.data.DataLoader, optim:torch.optim, loss:nn.Module) -> float:
    model.train()
    total_loss = .0
    scaler = GradScaler()

    for _, (l, x, y) in enumerate(dl):
        l = l.cuda()
        x = x.cuda()
        y = y.cuda()

        # Flatten and combine
        l = l.view(-1, 3*2*2)
        x = x.view(-1, 70*3*2*2)
        x = torch.cat((l, x), 1)

        y = y.view(-1, 70*2*2*2)

        # Forward pass
        with autocast():
            y_pred = model(x)
            l = loss(y_pred, y)
            total_loss += l.item()

        # Preform backpass
        scaler.scale(l).backward()
        scaler.step(optim)
        scaler.update()
    
    return total_loss / len(dl)


def eval(model:nn.Module, dl:torch.utils.data.DataLoader, loss:nn.Module) -> float:
    model.eval()
    total_loss = .0

    for _, (l, x, y) in enumerate(dl):
        l = l.cuda()
        x = x.cuda()
        y = y.cuda()

        # Flatten and combine
        l = l.view(-1, 3*2*2)
        x = x.view(-1, 70*3*2*2)
        x = torch.cat((l, x), 1)

        y = y.view(-1, 70*2*2*2)

        # Forward pass
        with autocast():
            y_pred = model(x)
            l = loss(y_pred, y)
            total_loss += l.item()

    return total_loss / len(dl)


def fold(data_file:str, model:nn.Module) -> tuple[float, float]:
    '''
    Do a cross fold training session with early stopping
    Inputs:
        - data_file (str): The file to load data from
        - model (nn.Module): The model to train
    '''
    loss_fn = nn.MSELoss()
    model = model.double().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # New Split  
    t_dl, v_dl = utils.get_dataloaders(data_file, 256, 0.7)

    # Early Stopping 
    eval_loss = -1*float('inf') 
    train_loss = 0 
    while train_loss > eval_loss:
        train_loss = train(model, t_dl, optimizer, loss_fn)
        eval_loss = eval(model, v_dl, loss_fn)

    return (train_loss, eval_loss)

if __name__ == '__main__':
    # Data 
    fname = join('/home/squirt/Documents/data/weather_data/', 'all_data.h5')

    # Do cross folds
    folds = 5 
    train_loss_rec = []
    eval_loss_rec = []
    for i in range(folds):
        model = simpleNN()
        train_loss, eval_loss = fold(fname, model)
        train_loss_rec.append(train_loss)
        eval_loss_rec.append(eval_loss)

    avg_train_loss = torch.mean(torch.tensor(train_loss_rec)).item()
    avg_eval_loss = torch.mean(torch.tensor(eval_loss_rec)).item()

    std_train_loss = torch.std(torch.tensor(train_loss_rec)).item() 
    std_eval_loss = torch.std(torch.tensor(eval_loss_rec)).item()

    print(f'Average Training Loss: {avg_train_loss} +/- {std_train_loss}') 
    print(f'Average Evaluation Loss: {avg_eval_loss} +/- {std_eval_loss}')
