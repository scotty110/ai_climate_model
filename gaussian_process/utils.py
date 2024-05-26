'''
Utility file, holds data prep, models, and eval functions.
Training loop is in main.py
'''
import torch
import h5py as h5

#-------------------------

'''
Data loading and processing functions
'''

def load_hdf5(filename:str):
    '''
    Load data from an HDF5 file and return a list of dictionaries.
    Inputs:
        - filename (str): Path to the HDF5 file.
    Outputs:
        - data (list): List of dictionaries, where each dictionary represents an entry in the original list.
    '''
    data = []  # List to hold dictionaries
    with h5.File(filename, 'r') as f:
        # Iterate through groups (each representing an entry in the original list)
        for group_name in f:
            group = f[group_name]
            # Reconstruct dictionary from datasets and attributes
            entry = {
                # Attributes (metadata)
                'day': group.attrs['day'],
                'region': group.attrs['region'],
                'time': group.attrs['time'],

                # groups (numpy arrays)
                'landmass': group['landmass'][...],  # Use [...] to read the full dataset
                'x': group['x'][...],
                'y': group['y'][...],
            }
            data.append(entry)
    return data


def stack_data(data:list[dict], key:str) -> torch.Tensor:
    return torch.stack([torch.tensor(entry[key]) for entry in data])


def generate_stacks(data:list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Create a PyTorch DataLoader from the data.
    Inputs:
        - data (list): List of dictionaries, where each dictionary represents an entry in the original list.
    Outputs:
        - landmass (torch.Tensor): Tensor of landmass data.
        - x (torch.Tensor): Tensor of x-coordinate data.
        - y (torch.Tensor): Tensor of y-coordinate data.
    '''
    landmass = stack_data(data, 'landmass')

    x = stack_data(data, 'x')
    x = x.transpose(2, 1)

    y = stack_data(data, 'y')
    y = y.transpose(2, 1)
    
    return (landmass, x, y)


def get_data(fname:str, split:int) -> tuple[tuple[torch.tensor], tuple[torch.tensor]]:
    '''
    Create PyTorch tensors to be used for training the GP
    Inputs:
        - fname (str): Path to the HDF5 file.
        - split (float): Fraction of the data to use for training.     
    Outputs:
        - Tuple of tuple of tensors to be used for training and validation.
    '''
    # Load data and create tensor 
    data = load_hdf5(fname)
    stacks = generate_stacks(data)

    # Split data into training and validation sets 
    indices = torch.randperm(stacks[0].size(0))

    # Shuffle
    landmass = stacks[0][indices]
    x = stacks[1][indices]
    y = stacks[2][indices]

    # Split
    split_idx = int(len(landmass) * split)

    # Training data
    train_data = (landmass[:split_idx], x[:split_idx], y[:split_idx])
    val_data = (landmass[split_idx:], x[split_idx:], y[split_idx:])
    return train_data, val_data